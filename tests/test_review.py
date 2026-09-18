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
    ReviewState(status=S.WAITING, oral_attempts=1),
    ReviewState(status=S.CHANGES, oral_attempts=1),
    ReviewState(Stage.WRITTEN, S.CHANGES, 1, 0),
    ReviewState(Stage.WRITTEN, S.WAITING, 1, 2),
    ReviewState(Stage.WRITTEN, S.ACCEPTED, 1, 2),
    ReviewState(status=S.FAILED, oral_attempts=3, written_attempts=2),
]


@pytest.mark.parametrize('state', STATES)
def test_failed_tests_and_rendering_do_not_mutate(state):
    assert step(state, E.TESTS_FAILED) is state
    assert ReviewState.from_columns(*state.columns()) == state


@pytest.mark.parametrize('state', [s for s in STATES if s.status != S.WAITING])
@pytest.mark.parametrize('event', MANUAL_REVIEW_EVENTS)
def test_manual_actions_require_waiting(state, event):
    before = replace(state)
    with pytest.raises(ValueError, match='waiting'):
        step(state, event)
    assert state == before


def test_oral_acceptance_is_rejected():
    with pytest.raises(ValueError, match='Written review'):
        step(ReviewState(status=S.WAITING, oral_attempts=1), E.ACCEPT)


@pytest.mark.parametrize('stage', list(Stage))
def test_oral_limit_when_scheduling(stage):
    state = ReviewState(stage, S.WAITING, 3, 2)
    failed = step(state, E.CHANGES_ORAL)
    assert failed.columns() == ('g3', 'o2')
    assert step(failed, E.TESTS_PASSED) == failed
    assert step(state, E.CHANGES_WRITTEN).status == S.CHANGES


def test_oral_limit_when_entering_legacy_or_corrupt_changes():
    state = ReviewState(status=S.CHANGES, oral_attempts=3)
    assert step(state, E.TESTS_PASSED).columns() == ('g3', 'o0')


@pytest.mark.parametrize('oral,written', [
    ('?1', '?1'), ('+1', '0'), ('0', '#0'), ('g3', '0'), ('3', 'o1'),
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
