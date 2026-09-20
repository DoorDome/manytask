from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from cachelib import SimpleCache

from manytask.rating_table import RatingTable
from manytask.review import ReviewEvent as E, ReviewStatus as S
from tests.sheets import Workbook, course_config, student


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def table():
    workbook = Workbook()
    ws = workbook.add_worksheet('main', rows=100, cols=3)
    ws.rows = [[], [], ['git', 'login', 'name'], []]
    cache = SimpleCache()
    config = course_config()
    cache.set('__config__', config.model_dump())
    table = RatingTable(ws, cache)
    table.sync_columns(config.deadlines)
    workbook.batch_update.reset_mock()
    return table


def submit(table, event=E.TESTS_PASSED, **kwargs):
    return table.store_score(student(), 'task', lambda _: 10, event, oral_attempt_limit=3, group_name="group", at=NOW, **kwargs)


def test_fresh_course_schema_and_complete_workflow(table):
    assert table.ws.row_values(4)[3:] == ['score', 'oral', 'written', 'reviewer'] * 2
    assert submit(table).review == '#'
    assert table.ws.rows[4][4:6] == ['#0', '0']
    assert submit(table, has_merge_request=True).review == '?'
    assert submit(table, E.CHANGES_WRITTEN).review == '-'
    assert submit(table).review == '?'
    assert submit(table, E.ACCEPT).review == '+'
    assert table.ws.rows[4][3:7] == ['10', '1', '+1', '']
    assert table.get_reviews('alice')['task'] == S.ACCEPTED
    before = deepcopy(table.ws.rows)
    with pytest.raises(ValueError):
        submit(table, E.ACCEPT)
    assert table.ws.rows == before


@pytest.mark.parametrize('event', [E.TESTS_FAILED, E.ACCEPT, E.CHANGES_WRITTEN])
def test_invalid_and_failed_events_do_not_create_student(table, event):
    before = deepcopy(table.ws.rows)
    if event == E.TESTS_FAILED:
        assert submit(table, event).review == ''
    else:
        with pytest.raises(ValueError):
            submit(table, event)
    assert table.ws.rows == before
    table.ws.append_row.assert_not_called()
    table.ws.spreadsheet.batch_update.assert_not_called()


def test_conflicting_main_state_is_rejected_without_writes(table):
    submit(table, has_merge_request=True)
    table.ws.rows[4][5] = '+1'
    table.ws.spreadsheet.batch_update.reset_mock()
    with pytest.raises(ValueError, match='Conflicting'):
        submit(table)
    table.ws.spreadsheet.batch_update.assert_not_called()


def test_repeated_submission_preserves_zero_score_and_other_tasks_on_cache_miss(table):
    table.store_score(student(), 'other', lambda _: 7, E.TESTS_PASSED, oral_attempt_limit=3, group_name="group", at=NOW)
    table.store_score(student(), 'task', lambda _: 0, E.TESTS_PASSED, oral_attempt_limit=3, group_name="group", at=NOW)
    table._cache.delete(f'{table.ws.id}:scores:alice')
    table._cache.delete(f'{table.ws.id}:reviews:alice')
    assert submit(table, has_merge_request=True).score == 0
    assert table.get_scores('alice') == {'task': 0, 'other': 7}
    assert table.get_reviews('alice') == {'task': S.READY_TO_BE_CHECKED, 'other': S.SOLVED_WITHOUT_MR}


def test_manual_action_does_not_regrade_or_reassign(table):
    table.update_reviewers_list(['assistant', 'second'])
    submit(table, has_merge_request=True)
    assert table.ws.rows[4][6] == 'assistant'
    update = Mock(side_effect=AssertionError('Manual regrade'))
    table.store_score(student(), 'task', update, E.CHANGES_WRITTEN, oral_attempt_limit=3, group_name="group", at=NOW)
    update.assert_not_called()
    assert table.ws.rows[4][3] == '10'
    assert table.ws.rows[4][6] == 'assistant'


def test_full_cache_includes_first_student_and_ignores_trailing_formula(table):
    submit(table, has_merge_request=True)
    table.ws.rows[4] += [''] * (table.ws.col_count - len(table.ws.rows[4])) + ['=SUM(D5:K5)']
    table.update_cached_scores()
    assert table.get_reviews('alice') == {'task': S.READY_TO_BE_CHECKED}
    assert table.get_scores('alice') == {'task': 10}
    assert table.get_all_scores_reviews()['alice']['task'] == (10, S.READY_TO_BE_CHECKED, None)


@pytest.mark.parametrize('status,expected', [
    (S.EMPTY, None), (S.SOLVED_WITHOUT_MR, None), (S.READY_TO_BE_CHECKED, None),
    (S.CHANGES_REQUESTED, False), (S.FAILED, False), (S.ACCEPTED, True),
])
def test_web_status(status, expected):
    from manytask.web import format_review_status
    assert format_review_status(status) is expected
