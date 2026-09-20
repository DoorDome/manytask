from dataclasses import replace

import pytest
from pydantic import ValidationError

from manytask.config import ManytaskDeadlinesConfig
from manytask.review import MANUAL_REVIEW_EVENTS, ReviewEvent as E, ReviewStage as Stage, ReviewState, ReviewStatus as S
from manytask.review import transition


def step(state, event, **kwargs):
    return transition(state, event, 3, **kwargs)


def test_complete_review_with_return_to_oral():
    state = step(ReviewState(), E.TESTS_PASSED)
    assert state.columns() == ('#0', '0')
    assert step(state, E.TESTS_PASSED) == state
    state = step(state, E.TESTS_PASSED, has_merge_request=True)
    assert state.columns() == ('?1', '0')
    assert step(state, E.TESTS_PASSED) == state
    state = step(state, E.CHANGES_ORAL)
    assert state.columns() == ('-1', '0')
    state = step(state, E.TESTS_PASSED)
    assert state.columns() == ('?2', '0')
    state = step(state, E.CHANGES_WRITTEN)
    assert state.columns() == ('2', '-0')
    state = step(state, E.TESTS_PASSED)
    assert state.columns() == ('2', '?1')
    state = step(state, E.CHANGES_WRITTEN)
    state = step(state, E.TESTS_PASSED)
    assert state.columns() == ('2', '?2')
    state = step(state, E.CHANGES_ORAL)
    state = step(state, E.TESTS_PASSED)
    assert state.columns() == ('?3', '2')
    state = step(state, E.CHANGES_WRITTEN)
    state = step(state, E.TESTS_PASSED)
    state = step(state, E.ACCEPT)
    assert state.columns() == ('3', '+3')
    assert step(state, E.TESTS_PASSED) == state


STATES = [
    ReviewState(), ReviewState(status=S.SOLVED_WITHOUT_MR),
    ReviewState(status=S.READY_TO_BE_CHECKED, oral_attempts=1),
    ReviewState(status=S.CHANGES_REQUESTED, oral_attempts=1),
    ReviewState(Stage.WRITTEN, S.CHANGES_REQUESTED, 1, 0),
    ReviewState(Stage.WRITTEN, S.READY_TO_BE_CHECKED, 1, 2),
    ReviewState(Stage.WRITTEN, S.ACCEPTED, 1, 2),
    ReviewState(status=S.FAILED, oral_attempts=3, written_attempts=2),
]


@pytest.mark.parametrize('state', STATES)
def test_failed_tests_and_rendering_do_not_mutate(state):
    assert step(state, E.TESTS_FAILED) is state
    assert ReviewState.from_columns(*state.columns()) == state


@pytest.mark.parametrize('state', [s for s in STATES if s.status != S.READY_TO_BE_CHECKED])
@pytest.mark.parametrize('event', MANUAL_REVIEW_EVENTS)
def test_manual_actions_require_ready_to_be_checked(state, event):
    before = replace(state)
    with pytest.raises(ValueError, match='waiting'):
        step(state, event)
    assert state == before


def test_oral_acceptance_is_allowed():
    state = ReviewState(status=S.READY_TO_BE_CHECKED, oral_attempts=1)
    accepted = step(state, E.ACCEPT)
    assert accepted.columns() == ('+1', '0')
    assert ReviewState.from_columns(*accepted.columns()) == accepted


@pytest.mark.parametrize('stage', list(Stage))
def test_oral_limit_when_scheduling(stage):
    state = ReviewState(stage, S.READY_TO_BE_CHECKED, 3, 2)
    failed = step(state, E.CHANGES_ORAL)
    assert failed.columns() == ('g3', 'o2')
    assert step(failed, E.TESTS_PASSED) == failed
    assert step(state, E.CHANGES_WRITTEN).status == S.CHANGES_REQUESTED


def test_enter_review_does_not_recheck_changed_limit():
    state = ReviewState(status=S.CHANGES_REQUESTED, oral_attempts=3)
    assert step(state, E.TESTS_PASSED).columns() == ('?4', '0')


@pytest.mark.parametrize('oral,written', [
    ('?1', '?1'), ('+0', '0'), ('0', '#1'), ('g3', '0'), ('3', 'o1'),
    ('#1', '0'), ('1', '2'), ('?', '0'), ("'?1", '0'), ('-x', '0'),
    ('?0', '0'), ('1', '+0'), ('-1', '-1'), ('-2.5', '0'),
])
def test_invalid_columns(oral, written):
    with pytest.raises(ValueError):
        ReviewState.from_columns(oral, written)


def test_empty_cells():
    assert ReviewState.from_columns('', '') == ReviewState()


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '3'])
def test_strict_limit(limit):
    with pytest.raises(ValidationError):
        ManytaskDeadlinesConfig(timezone='UTC', schedule=[], oral_attempt_limit=limit)
    with pytest.raises(ValueError):
        transition(ReviewState(), E.TESTS_PASSED, limit)


def test_default_limit_and_first_submission_with_mr():
    config = ManytaskDeadlinesConfig(timezone='UTC', schedule=[])
    assert config.oral_attempt_limit == 3
    assert step(ReviewState(), E.TESTS_PASSED, has_merge_request=True).columns() == ('?1', '0')


@pytest.mark.parametrize('event', [True, False, 'approve', 'unknown'])
def test_event_contract_is_explicit(event):
    with pytest.raises(ValueError, match='ReviewEvent'):
        step(ReviewState(), event)


@pytest.mark.parametrize('oral,written,expected', [
    ('-1', '2', ('?2', '2')),
    ('2', '-0', ('2', '?1')),
    ('3', '-5', ('3', '?6')),
    ('?1', '2', ('?1', '2')),
    ('2', '?3', ('2', '?3')),
    ('2', '+3', ('2', '+3')),
    ('g3', 'o2', ('g3', 'o2')),
])
@pytest.mark.parametrize('has_merge_request', [False, True])
def test_passing_report_preserves_stage_and_counts(oral, written, expected, has_merge_request):
    state = ReviewState.from_columns(oral, written)
    assert step(state, E.TESTS_PASSED, has_merge_request=has_merge_request).columns() == expected
    assert state.columns() == (oral, written)


@pytest.mark.parametrize('stages', [(Stage.ORAL,), (Stage.WRITTEN,),
                                    (Stage.ORAL, Stage.WRITTEN), (Stage.WRITTEN, Stage.ORAL)])
def test_configured_pipeline_first_stage_and_immediate_acceptance(stages):
    state = step(ReviewState(), E.TESTS_PASSED, review_stages=stages)
    assert state.stage == stages[0]
    assert state.status == S.SOLVED_WITHOUT_MR
    assert ReviewState.from_columns(*state.columns()) == state
    state = step(state, E.TESTS_PASSED, review_stages=stages, has_merge_request=True)
    assert state.stage == stages[0]
    assert (state.oral_attempts, state.written_attempts) == ((1, 0) if stages[0] == Stage.ORAL else (0, 1))
    accepted = step(state, E.ACCEPT, review_stages=stages)
    assert accepted.status == S.ACCEPTED
    assert accepted.stage == stages[0]
    assert ReviewState.from_columns(*accepted.columns()) == accepted
    assert step(accepted, E.TESTS_PASSED, review_stages=stages) == accepted


@pytest.mark.parametrize('stage,event', [(Stage.ORAL, E.CHANGES_WRITTEN), (Stage.WRITTEN, E.CHANGES_ORAL)])
def test_disabled_target_stage_is_rejected(stage, event):
    state = ReviewState(stage, S.READY_TO_BE_CHECKED, 1, 1)
    with pytest.raises(ValueError, match='disabled'):
        step(state, event, review_stages=(stage,))


@pytest.mark.parametrize('status,event', [(S.READY_TO_BE_CHECKED, E.ACCEPT),
                                         (S.CHANGES_REQUESTED, E.TESTS_PASSED)])
def test_disabling_current_stage_requires_explicit_state_resolution(status, event):
    state = ReviewState(Stage.ORAL, status, 1, 0)
    with pytest.raises(ValueError, match='Current review stage'):
        step(state, event, review_stages=(Stage.WRITTEN,))
    assert step(state, E.TESTS_FAILED, review_stages=(Stage.WRITTEN,)) == state


@pytest.mark.parametrize('stages', [[], ['oral', 'oral'], ['written', 'written'], ['unknown'],
                                    ['oral', 'written', 'oral'], None, 'oral'])
def test_invalid_task_review_configuration(stages):
    from manytask.config import ManytaskTaskConfig
    with pytest.raises(ValidationError):
        ManytaskTaskConfig(task='task', score=10, review_stages=stages)


@pytest.mark.parametrize('stages', [['oral'], ['written'], ['oral', 'written'], ['written', 'oral']])
def test_task_review_configuration_roundtrip(stages):
    from manytask.config import ManytaskTaskConfig
    task = ManytaskTaskConfig(task='task', score=10, review_stages=stages)
    assert task.review_stages == tuple(Stage(s) for s in stages)
    assert ManytaskTaskConfig.model_validate_json(task.model_dump_json()) == task
