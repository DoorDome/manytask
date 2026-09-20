from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from flask import Flask

from manytask.review import MANUAL_REVIEW_EVENTS
from tests.sheets import course_config, student
from tests.test_review_storage import table  # noqa: F401


@pytest.fixture
def api(monkeypatch, table):
    monkeypatch.setenv('TESTER_TOKEN', 'test-token')
    from manytask import api as module
    monkeypatch.setattr(module, 'TESTER_TOKEN', 'test-token')
    config = course_config()
    app = Flask(__name__)
    app.config['TESTING'] = True
    gitlab = Mock()
    gitlab.get_student_by_username.side_effect = student
    gitlab.is_reviewer.side_effect = lambda s: s.username == 'assistant'
    gitlab.list_reviewers.return_value = ['assistant']
    app.course = SimpleNamespace(config=config, deadlines=config.deadlines, rating_table=table,
                                 gitlab_api=gitlab, solutions_api=Mock())
    app.register_blueprint(module.bp)
    return app.test_client(), app.course


def report(api, token='test-token', **data):
    client, _ = api
    return client.post('/api/report', headers={'Authorization': f'Bearer {token}'}, data={
        'task': 'task', 'username': 'alice', 'request_type': 'submit', **data,
    })


def test_api_end_to_end(api):
    assert report(api).json['review_status'] == '#'
    assert report(api, request_type='changes_written', reported_by='assistant').status_code == 409
    assert report(api, merge_request_iid='42').json['review_status'] == '?'
    assert report(api, request_type='changes_written', reported_by='assistant').json['review_status'] == '-'
    assert report(api).json['review_status'] == '?'
    assert report(api, request_type='approve', reported_by='assistant').json['review_status'] == '+'
    before = deepcopy(api[1].rating_table.ws.rows)
    assert report(api, request_type='approve', reported_by='assistant').status_code == 409
    assert api[1].rating_table.ws.rows == before


@pytest.mark.parametrize('mr', ['', ' ', '\t'])
def test_empty_mr_does_not_start_review(api, mr):
    assert report(api, merge_request_iid=mr).json['review_status'] == '#'
    api[1].rating_table.ws.spreadsheet.batch_update.reset_mock()
    assert report(api, request_type='changes_oral', reported_by='assistant').status_code == 409
    api[1].rating_table.ws.spreadsheet.batch_update.assert_not_called()
    api[1].gitlab_api.list_reviewers.assert_not_called()


@pytest.mark.parametrize('score', ['0', '5', '0.5'])
def test_partial_scores_do_not_start_review(api, score):
    assert report(api, score=score, merge_request_iid='12').json['review_status'] == ''
    api[1].rating_table.ws.spreadsheet.batch_update.assert_not_called()
    api[1].rating_table.ws.append_row.assert_not_called()


def test_success_is_evaluated_before_deadline_penalty(api):
    result = report(api, score='10', submit_time='2100-01-01 00:00:00+0000', merge_request_iid='12')
    assert result.status_code == 200
    assert result.json['review_status'] == '?'
    assert result.json['score'] == 0


@pytest.mark.parametrize('data,status', [
    ({'token': 'wrong'}, 403),
    ({'request_type': 'reject'}, 400),
    ({'request_type': 'approve'}, 400),
    ({'request_type': 'approve', 'reported_by': 'alice'}, 403),
    ({'request_type': 'tests_passed'}, 400),
    ({'request_type': 'tests_failed'}, 400),
])
def test_authentication_and_manual_request_validation(api, data, status):
    assert report(api, **data).status_code == status
    api[1].rating_table.ws.spreadsheet.batch_update.assert_not_called()


def test_oral_limit_in_api(api):
    api[1].deadlines.oral_attempt_limit = 1
    assert report(api, merge_request_iid='12').json['review_status'] == '?'
    assert report(api, request_type='changes_oral', reported_by='assistant').json['review_status'] == 'failed'
    assert report(api).json['review_status'] == 'failed'
    assert api[1].rating_table.ws.rows[4][4:6] == ['g1', 'o0']


def test_ci_exposes_exactly_three_manual_actions():
    from pathlib import Path
    config = yaml.safe_load(Path('ci/review.gitlab-ci.yml').read_text())
    jobs = {name: job for name, job in config.items() if not name.startswith('.')}
    assert set(jobs) == {'review-accept', 'review-changes-oral', 'review-changes-written'}
    assert {job['variables']['REVIEW_ACTION'] for job in jobs.values()} == {e.value for e in MANUAL_REVIEW_EVENTS}
    assert config['.manytask-review']['rules'][0]['when'] == 'manual'


def test_summary_failure_after_acceptance_is_still_success(api):
    assert report(api, merge_request_iid='12').status_code == 200
    assert report(api, request_type='changes_written', reported_by='assistant').status_code == 200
    assert report(api).status_code == 200
    api[1].rating_table.ws.spreadsheet.worksheet.side_effect = RuntimeError('summary unavailable')
    result = report(api, request_type='approve', reported_by='assistant')
    assert result.status_code == 200
    assert result.json['review_status'] == '+'
    assert api[1].rating_table.ws.rows[4][4:6] == ['1', '+1']


def test_api_preserves_instants_and_uses_server_time_for_manual_review(api, monkeypatch):
    from datetime import datetime, timezone
    from manytask.config import ManytaskDeadlinesConfig
    from manytask.review_sheet import REVIEW_DETAILS_COLUMNS
    now = datetime(2026, 1, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(ManytaskDeadlinesConfig, 'get_now_with_timezone', lambda _: now)
    result = report(api, submit_time='2026-01-01 03:00:00+0300', merge_request_iid='12')
    assert result.json['submit_time'] == '2026-01-01 00:00:00+00:00'
    report(api, request_type='changes_written', reported_by='assistant', submit_time='2000-01-01 00:00:00+0000')
    row = api[1].rating_table.ws.spreadsheet.worksheet('review_details').rows[1]
    values = dict(zip(REVIEW_DETAILS_COLUMNS, row))
    assert values['first_successful_submission_at'] == '2026-01-01 00:00:00+00:00'
    assert values['last_oral_review_at'] == now.isoformat(sep=' ')


@pytest.mark.parametrize('stages,first,accepted', [
    (['oral'], ['?1', '0'], ['+1', '0']),
    (['written'], ['0', '?1'], ['0', '+1']),
    (['oral', 'written'], ['?1', '0'], ['+1', '0']),
    (['written', 'oral'], ['0', '?1'], ['0', '+1']),
])
def test_task_pipeline_through_api_cache_and_summary(api, stages, first, accepted):
    from manytask.review import ReviewStage, ReviewStatus
    from manytask.review_sheet import REVIEW_DETAILS_COLUMNS
    course = api[1]
    task = course.deadlines.find_task('task')[1]
    task.review_stages = tuple(ReviewStage(stage) for stage in stages)
    assert report(api).json['review_status'] == '#'
    assert course.rating_table.ws.rows[4][4:6] == (['#0', '0'] if stages[0] == 'oral' else ['0', '#0'])
    assert report(api, merge_request_iid='42').status_code == 200
    assert course.rating_table.ws.rows[4][4:6] == first
    assert report(api, request_type='approve', reported_by='assistant').json['review_status'] == '+'
    assert course.rating_table.ws.rows[4][4:6] == accepted
    course.rating_table._cache.set('__config__', course.config.model_dump())
    course.rating_table.update_cached_scores()
    assert course.rating_table.get_reviews('alice')['task'] == ReviewStatus.ACCEPTED
    summary = course.rating_table.ws.spreadsheet.worksheet('review_details')
    values = dict(zip(REVIEW_DETAILS_COLUMNS, summary.rows[1]))
    assert values['stage'] == stages[0]
    assert values['status'] == '+'
    assert values[f'last_{stages[0]}_review_at']
    other = 'written' if stages[0] == 'oral' else 'oral'
    assert values[f'last_{other}_review_at'] == ''
    # A neighboring task still uses its own default pipeline.
    assert report(api, task='other', merge_request_iid='43').status_code == 200
    assert course.rating_table.ws.rows[4][8:10] == ['?1', '0']


@pytest.mark.parametrize('stage,forbidden', [('oral', 'changes_written'), ('written', 'changes_oral')])
def test_disabled_stage_api_error_has_no_writes(api, stage, forbidden):
    from manytask.review import ReviewStage
    api[1].deadlines.find_task('task')[1].review_stages = (ReviewStage(stage),)
    assert report(api, merge_request_iid='42').status_code == 200
    workbook = api[1].rating_table.ws.spreadsheet
    workbook.batch_update.reset_mock()
    result = report(api, request_type=forbidden, reported_by='assistant')
    assert result.status_code == 409
    assert 'disabled' in result.text
    workbook.batch_update.assert_not_called()
