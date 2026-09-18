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
    assert report(api, request_type='approve', reported_by='assistant').status_code == 409
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
