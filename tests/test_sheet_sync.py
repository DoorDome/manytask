from copy import deepcopy

import pytest
from cachelib import SimpleCache

from manytask.rating_table import RatingTable
from manytask.review import ReviewEvent
from tests.sheets import Workbook, course_config, student
from tests.test_review_storage import NOW


@pytest.fixture
def filled_table():
    workbook = Workbook()
    ws = workbook.add_worksheet('main', rows=100, cols=3)
    ws.rows = [[], [], ['git', 'login', 'name'], []]
    table = RatingTable(ws, SimpleCache())
    table.sync_columns(course_config([('first', ['a', 'b']), ('second', ['c'])]).deadlines)
    ws.rows.append(['link', 'alice', 'Alice', '7', '1', '+2', 'anna', '0', '?1', '0', 'boris',
                    '8', 'g3', 'o1', 'chris', '=SUM(D5:O5)'])
    ws.col_count += 1
    ws.formats[4, 3] = {'numberFormat': {'type': 'NUMBER'}}
    workbook.batch_update.reset_mock()
    return table


def snapshot_tasks(table):
    return {name: table.ws.rows[4][column - 1:column + 3]
            for column, name in table._list_tasks(with_index=True) if name}


@pytest.mark.parametrize('groups', [
    [('first', ['new', 'a', 'b']), ('second', ['c'])],
    [('first', ['a', 'new', 'b']), ('second', ['c'])],
    [('first', ['a', 'b', 'new']), ('second', ['c'])],
    [('first', ['a', 'b']), ('second', ['new', 'c'])],
    [('first', ['a', 'b']), ('second', ['c', 'new'])],
    [('new-group', ['new']), ('first', ['a', 'b']), ('second', ['c'])],
    [('first', ['a', 'b']), ('new-group', ['new']), ('second', ['c'])],
    [('first', ['a', 'b']), ('second', ['c']), ('new-group', ['new'])],
])
def test_insert_preserves_data_order_headers_and_is_idempotent(filled_table, groups):
    table = filled_table
    before = snapshot_tasks(table)
    config = course_config(groups)
    table.sync_columns(config.deadlines)
    assert [name for _, name in table._list_tasks(with_index=True) if name] == [t.name for t in config.get_tasks()]
    after = snapshot_tasks(table)
    assert {name: after[name] for name in before} == before
    assert after['new'] == ['', '', '', '']
    assert table.ws.rows[4][-1] == '=SUM(D5:O5)'
    assert table.ws.formats[4, table._find_task_column('a') - 1] == {'numberFormat': {'type': 'NUMBER'}}
    expected_labels = [label for group, names in groups for label in [group, *[''] * (len(names) - 1)]]
    assert [table.ws.rows[0][column - 1] for column, name in table._list_tasks(with_index=True) if name] == expected_labels
    before_repeat = deepcopy(table.ws.rows)
    table.ws.spreadsheet.batch_update.reset_mock()
    table.sync_columns(config.deadlines)
    assert table.ws.rows == before_repeat
    table.ws.spreadsheet.batch_update.assert_not_called()
    table.ws.spreadsheet.worksheet.assert_not_called()
    table.ws.resize.assert_not_called()
    table.ws.update_cells.assert_not_called()


def test_multiple_insertions_do_not_rewrite_unrelated_headers(filled_table):
    table = filled_table
    config = course_config([('first', ['new1', 'a', 'new2', 'b']), ('second', ['c']), ('third', ['d', 'e'])])
    before = snapshot_tasks(table)
    table.sync_columns(config.deadlines)
    after = snapshot_tasks(table)
    assert {name: after[name] for name in before} == before
    bodies = table.ws.spreadsheet.batch_update.call_args_list
    assert len(bodies) == 1
    old_starts = {table._find_task_column(name) - 1 for name in before}
    updates = [r['updateCells'] for r in bodies[0].args[0]['requests'] if 'updateCells' in r]
    for update in updates:
        start = update['start']
        if start['columnIndex'] in old_starts:
            assert start['rowIndex'] == 0
            assert start['columnIndex'] == table._find_task_column('a') - 1
        assert start['rowIndex'] < 4
    formats = [r['repeatCell']['range'] for r in bodies[0].args[0]['requests'] if 'repeatCell' in r]
    assert all(bounds['startColumnIndex'] not in old_starts for bounds in formats)
    assert snapshot_tasks(table)['new1'] == ['', '', '', '']


def test_new_student_uses_inserted_schema(filled_table):
    table = filled_table
    table.sync_columns(course_config([('first', ['new', 'a', 'b']), ('second', ['c'])]).deadlines)
    status = table.store_score(student('bob'), 'new', lambda _: 10, ReviewEvent.TESTS_PASSED,
                               oral_attempt_limit=3, group_name='first', at=NOW, has_merge_request=True)
    assert status.review == '?'
    assert table.ws.rows[5][3:7] == ['10', '?1', '0', '']
    assert table._read_task_values(table.ws.rows[5], table._find_task_column('a'))[0] is None


@pytest.mark.parametrize('groups', [
    [('first', ['b', 'a']), ('second', ['c'])],
    [('first', ['a']), ('second', ['c'])],
    [('first', ['a']), ('second', ['b', 'c'])],
])
def test_unsupported_reorder_removal_or_move_fails_before_writes(filled_table, groups):
    before = deepcopy(filled_table.ws.rows)
    with pytest.raises(ValueError):
        filled_table.sync_columns(course_config(groups).deadlines)
    assert filled_table.ws.rows == before
    filled_table.ws.spreadsheet.batch_update.assert_not_called()


def test_blank_course_and_repeated_sync():
    workbook = Workbook()
    ws = workbook.add_worksheet('main', rows=100, cols=3)
    table = RatingTable(ws, SimpleCache())
    config = course_config([('first', ['a']), ('second', ['b'])])
    table.sync_columns(config.deadlines)
    assert table._find_task_column('a') == 4
    assert table._find_task_column('b') == 8
    assert ws.rows[3][3:11] == ['score', 'oral', 'written', 'reviewer'] * 2
    workbook.batch_update.reset_mock()
    table.sync_columns(config.deadlines)
    workbook.batch_update.assert_not_called()


def test_sync_preserves_separate_summary(filled_table):
    table = filled_table
    summary = table.ws.spreadsheet.add_worksheet('review_details', rows=10, cols=11)
    summary.rows = [['historical data']]
    table.sync_columns(course_config([('first', ['new', 'a', 'b']), ('second', ['c'])]).deadlines)
    assert summary.rows == [['historical data']]
    table.ws.spreadsheet.worksheet.assert_not_called()


@pytest.mark.parametrize('corruption', ['duplicate', 'gap'])
def test_invalid_task_layout_is_rejected_without_changes(filled_table, corruption):
    table = filled_table
    if corruption == 'duplicate':
        table.ws.rows[2][7] = 'a'
    else:
        table.ws.rows[2][7] = ''
    before = deepcopy(table.ws.rows)
    with pytest.raises(ValueError):
        table.sync_columns(course_config([('first', ['a', 'b']), ('second', ['c', 'new'])]).deadlines)
    assert table.ws.rows == before
    table.ws.spreadsheet.batch_update.assert_not_called()


def configure_stages(config, name, *stages):
    from manytask.review import ReviewStage
    config.deadlines.find_task(name)[1].review_stages = tuple(ReviewStage(stage) for stage in stages)


def test_mixed_pipelines_hide_only_unused_columns_and_preserve_four_column_blocks():
    workbook = Workbook()
    ws = workbook.add_worksheet("main", rows=100, cols=3)
    table = RatingTable(ws, SimpleCache())
    config = course_config([('first', ['oral', 'written', 'both'])])
    configure_stages(config, 'oral', 'oral')
    configure_stages(config, 'written', 'written')
    configure_stages(config, 'both', 'written', 'oral')
    table.sync_columns(config.deadlines)
    assert ws.col_count == 15
    assert ws.hidden_columns == {5, 8}  # F: oral task's written; I: written task's oral.
    assert [table._find_task_column(name) for name in ('oral', 'written', 'both')] == [4, 8, 12]
    assert ws.rows[3][3:] == ['score', 'oral', 'written', 'reviewer'] * 3
    workbook.batch_update.reset_mock()
    table.sync_columns(config.deadlines)
    workbook.batch_update.assert_not_called()


def test_existing_pipeline_changes_only_visibility_and_can_unhide(filled_table):
    table = filled_table
    config = course_config([('first', ['a', 'b']), ('second', ['c'])])
    configure_stages(config, 'a', 'oral')
    configure_stages(config, 'b', 'written')
    before = deepcopy(table.ws.rows)
    width = table.ws.col_count
    table.sync_columns(config.deadlines)
    assert table.ws.hidden_columns == {5, 8}
    assert table.ws.rows == before
    assert table.ws.col_count == width
    requests = table.ws.spreadsheet.batch_update.call_args.args[0]['requests']
    assert all('updateDimensionProperties' in request for request in requests)
    configure_stages(config, 'a', 'written', 'oral')
    configure_stages(config, 'b', 'oral', 'written')
    table.sync_columns(config.deadlines)
    assert table.ws.hidden_columns == set()
    assert table.ws.rows == before
    table.ws.spreadsheet.batch_update.reset_mock()
    table.sync_columns(config.deadlines)
    table.ws.spreadsheet.batch_update.assert_not_called()


def test_insertions_shift_hidden_columns_and_reset_inherited_visibility(filled_table):
    table = filled_table
    config = course_config([('first', ['a', 'b']), ('second', ['c'])])
    configure_stages(config, 'a', 'oral')
    configure_stages(config, 'b', 'written')
    table.sync_columns(config.deadlines)
    # Inserting after a manually hidden reviewer column must not hide a new task.
    table.ws.hidden_columns.add(6)
    before = snapshot_tasks(table)
    config = course_config([('first', ['new1', 'a', 'new2', 'b']), ('second', ['c'])])
    configure_stages(config, 'a', 'oral')
    configure_stages(config, 'b', 'written')
    configure_stages(config, 'new1', 'written')
    table.sync_columns(config.deadlines)
    assert table.ws.hidden_columns == {4, 9, 10, 16}
    assert {name: snapshot_tasks(table)[name] for name in before} == before
    assert table.ws.rows[4][-1] == '=SUM(D5:O5)'
    table.ws.spreadsheet.batch_update.reset_mock()
    table.sync_columns(config.deadlines)
    table.ws.spreadsheet.batch_update.assert_not_called()


def test_column_metadata_start_offset_and_other_sheets(filled_table):
    table = filled_table
    table.ws.hidden_columns = {5}
    table.ws.spreadsheet.fetch_sheet_metadata.return_value = {
        'sheets': [
            {'properties': {'sheetId': 999}, 'data': [{'columnMetadata': [{'hiddenByUser': True}]}]},
            {'properties': {'sheetId': table.ws.id}, 'data': [
                {'startColumn': 4, 'columnMetadata': [{}, {'hiddenByUser': True}]},
            ]},
        ],
    }
    table.ws.spreadsheet.fetch_sheet_metadata.side_effect = None
    config = course_config([('first', ['a', 'b']), ('second', ['c'])])
    configure_stages(config, 'a', 'oral')
    table.sync_columns(config.deadlines)
    table.ws.spreadsheet.batch_update.assert_not_called()
    params = table.ws.spreadsheet.fetch_sheet_metadata.call_args.kwargs['params']
    assert params['ranges'] == "'main'"
    assert 'columnMetadata(hiddenByUser)' in params['fields']
