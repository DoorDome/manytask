from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from manytask.review import ReviewEvent as E, ReviewStage as Stage, ReviewState, ReviewStatus as S
from manytask.review_sheet import REVIEW_DETAILS_COLUMNS, ReviewDetailsSheet, ReviewTimestamps, parse_timestamp
from tests.sheets import Workbook, student
from tests.test_review_storage import NOW, submit, table  # noqa: F401


def dates_row(table):
    return dict(zip(REVIEW_DETAILS_COLUMNS, table.ws.spreadsheet.worksheet('review_details').rows[1]))


def store_at(table, event, at):
    return table.store_score(student(), 'task', lambda _: 10, event,
                             oral_attempt_limit=3, group_name='group', at=at, has_merge_request=True)


def test_dates_follow_completed_stage_and_submission_time(table):
    submit(table)  # first successful submission precedes the MR
    assert dates_row(table)['first_successful_submission_at'] == NOW.isoformat(sep=' ')
    later = NOW + timedelta(hours=1)
    store_at(table, E.TESTS_PASSED, later)
    store_at(table, E.CHANGES_WRITTEN, later + timedelta(minutes=1))
    assert dates_row(table)['last_oral_review_at'] == (later + timedelta(minutes=1)).isoformat(sep=' ')
    assert dates_row(table)['last_written_review_at'] == ''
    store_at(table, E.TESTS_PASSED, later + timedelta(minutes=2))
    store_at(table, E.ACCEPT, later + timedelta(minutes=3))
    saved = dates_row(table)
    assert saved['last_written_review_at'] == (later + timedelta(minutes=3)).isoformat(sep=' ')
    assert saved['first_successful_submission_at'] == NOW.isoformat(sep=' ')
    store_at(table, E.TESTS_PASSED, NOW - timedelta(days=1))
    assert dates_row(table)['first_successful_submission_at'] == saved['first_successful_submission_at']
    assert dates_row(table)['last_successful_submission_at'] == (NOW - timedelta(days=1)).isoformat(sep=' ')
    assert dates_row(table)['status'] == '+'
    assert len(table.ws.spreadsheet.worksheet('review_details').rows) == 2


def test_stale_summary_never_controls_transition(table):
    submit(table, has_merge_request=True)
    details = table.ws.spreadsheet.worksheet('review_details')
    for index in [3, 5, 9, 10]:
        details.rows[1][index] = 'corrupt'
    assert submit(table, E.CHANGES_WRITTEN).review == '-'
    assert table.ws.rows[4][4:6] == ['1', '-0']
    assert dates_row(table)['stage'] == 'written'
    assert dates_row(table)['oral_attempts'] == '1'
    assert dates_row(table)['last_oral_review_at'] == NOW.isoformat(sep=' ')


def test_missing_summary_does_not_reset_attempts(table):
    submit(table, has_merge_request=True)
    workbook = table.ws.spreadsheet
    del workbook.sheets['review_details']
    assert submit(table, E.CHANGES_ORAL).review == '-'
    assert submit(table).review == '?'
    assert table.ws.rows[4][4:6] == ['?2', '0']


def test_main_write_precedes_separate_summary_write(table):
    submit(table)
    batches = table.ws.spreadsheet.batch_update.call_args_list
    assert len(batches) == 2
    assert batches[0].args[0]['requests'][0]['updateCells']['start']['sheetId'] == table.ws.id
    assert batches[1].args[0]['requests'][0]['updateCells']['start']['sheetId'] != table.ws.id


@pytest.mark.parametrize('event', [E.TESTS_FAILED, E.ACCEPT, E.CHANGES_ORAL])
def test_no_summary_access_for_invalid_or_failed_event(table, event):
    workbook = table.ws.spreadsheet
    workbook.worksheet.reset_mock()
    workbook.add_worksheet.reset_mock()
    if event == E.TESTS_FAILED:
        submit(table, event)
    else:
        with pytest.raises(ValueError):
            submit(table, event)
    workbook.worksheet.assert_not_called()
    workbook.add_worksheet.assert_not_called()
    workbook.batch_update.assert_not_called()


def test_main_write_failure_does_not_access_summary(table):
    workbook = table.ws.spreadsheet
    workbook.batch_update.side_effect = RuntimeError('main unavailable')
    with pytest.raises(RuntimeError, match='main unavailable'):
        submit(table)
    workbook.worksheet.assert_not_called()


@pytest.mark.parametrize('failure', ['unavailable', 'bad_date', 'write_error'])
def test_summary_failure_does_not_fail_main_transition(table, failure, caplog):
    submit(table, has_merge_request=True)
    workbook = table.ws.spreadsheet
    details = workbook.worksheet('review_details')
    if failure == 'unavailable':
        workbook.worksheet.side_effect = RuntimeError('summary unavailable')
    elif failure == 'bad_date':
        details.rows[1][7] = 'bad date'
    else:
        def fail_summary(body):
            if body['requests'][0]['updateCells']['start']['sheetId'] != table.ws.id:
                raise RuntimeError('summary write failed')
            workbook._batch_update(body)
        workbook.batch_update.side_effect = fail_summary
    assert submit(table, E.CHANGES_WRITTEN).review == '-'
    assert table.ws.rows[4][4:6] == ['1', '-0']
    assert 'Cannot update review_details for alice/task' in caplog.text


def test_sheet_upsert_survives_recreation_and_expands():
    workbook = Workbook()
    summary = ReviewDetailsSheet(workbook)
    summary.ws.row_count = 1
    state = ReviewState(status=S.SOLVED_WITHOUT_MR)
    summary.record('alice', 'task', 'group', ReviewState(), state, E.TESTS_PASSED, NOW)
    assert summary.ws.row_count == 2
    summary = ReviewDetailsSheet(workbook)
    summary.record('alice', 'task', 'group', state, state, E.TESTS_PASSED, NOW + timedelta(hours=1))
    assert len(summary.ws.rows) == 2
    workbook.add_worksheet.assert_called_once()
    summary.ws.update_cells.assert_called_once()
    summary.record('bob', 'task', 'group', ReviewState(), state, E.TESTS_PASSED, NOW)
    assert len(summary.ws.rows) == 3


def test_duplicate_key_and_wrong_header_are_not_overwritten():
    workbook = Workbook()
    summary = ReviewDetailsSheet(workbook)
    summary.ws.rows.extend([['alice', 'task'], ['alice', 'task']])
    snapshot = deepcopy(summary.ws.rows)
    with pytest.raises(ValueError, match='Duplicate'):
        summary.record('alice', 'task', 'group', ReviewState(), ReviewState(), E.TESTS_PASSED, NOW)
    assert summary.ws.rows == snapshot
    summary.ws.rows[0][0] = 'unexpected'
    with pytest.raises(ValueError, match='schema'):
        ReviewDetailsSheet(workbook)
    assert summary.ws.rows[0][0] == 'unexpected'


@pytest.mark.parametrize('value', ['2026-01-01 12:00:00', 'not a timestamp'])
def test_invalid_timestamps(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_empty_and_aware_dates():
    assert parse_timestamp('') is None
    assert parse_timestamp('2026-01-01 03:00:00+03:00') == NOW
    with pytest.raises(ValueError):
        ReviewTimestamps().updated(E.TESTS_PASSED, Stage.ORAL, datetime(2026, 1, 1))
    dates = ReviewTimestamps(first_successful_submission_at=NOW)
    assert dates.updated(E.TESTS_FAILED, Stage.ORAL, NOW) is dates


def test_write_requests_only_builds_requests():
    workbook = Workbook()
    summary = ReviewDetailsSheet(workbook)
    requests = summary.write_requests(2, 'alice', 'task', 'group', ReviewState(), ReviewTimestamps())
    assert requests[0]['updateCells']['start'] == {'sheetId': summary.ws.id, 'rowIndex': 1, 'columnIndex': 0}
    workbook.batch_update.assert_not_called()


def test_invalid_main_state_never_reads_summary(table):
    submit(table, has_merge_request=True)
    table.ws.rows[4][5] = '?2'
    table.ws.spreadsheet.worksheet.reset_mock()
    with pytest.raises(ValueError):
        submit(table)
    table.ws.spreadsheet.worksheet.assert_not_called()
