"""In-memory Sheets values with recorded requests; no live Google credentials."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import gspread
from gspread.utils import rowcol_to_a1

from manytask.config import ManytaskConfig


def course_config(groups=None):
    if groups is None:
        groups = [('group', ['task', 'other'])]
    return ManytaskConfig(
        version=1,
        settings=dict(course_name='test', gitlab_base_url='https://gitlab.test',
                      public_repo='course/public', students_group='course/students'),
        ui=dict(task_url_template='https://gitlab.test/$TASK_NAME'),
        deadlines=dict(timezone='UTC', schedule=[
            dict(group=name, start='2000-01-01T00:00:00+00:00', end='2099-01-01T00:00:00+00:00',
                 tasks=[dict(task=task, score=10) for task in tasks])
            for name, tasks in groups
        ]),
    )


def student(username='alice'):
    return SimpleNamespace(id=1, username=username, name='Alice', repo=f'https://gitlab.test/{username}')


class Workbook:
    def __init__(self):
        self.sheets = {}
        self.batch_update = Mock(side_effect=self._batch_update)
        self.worksheet = Mock(side_effect=self._worksheet)
        self.add_worksheet = Mock(side_effect=self._add_worksheet)

    def _worksheet(self, title):
        if title not in self.sheets:
            raise gspread.WorksheetNotFound(title)
        return self.sheets[title]

    def _add_worksheet(self, title, rows, cols):
        sheet = Sheet(self, title, len(self.sheets) + 1, rows=rows, cols=cols)
        self.sheets[title] = sheet
        return sheet

    def _batch_update(self, body):
        for request in body['requests']:
            if 'updateCells' in request:
                update = request['updateCells']
                start = update['start']
                sheet = next(s for s in self.sheets.values() if s.id == start['sheetId'])
                for offset, row in enumerate(update['rows']):
                    for col, cell in enumerate(row['values']):
                        value = next(iter(cell['userEnteredValue'].values()))
                        sheet.set_cell(start['rowIndex'] + offset, start['columnIndex'] + col, str(value))
            elif 'insertDimension' in request:
                bounds = request['insertDimension']['range']
                sheet = next(s for s in self.sheets.values() if s.id == bounds['sheetId'])
                assert bounds['dimension'] == 'COLUMNS'
                start, end = bounds['startIndex'], bounds['endIndex']
                assert start <= sheet.col_count
                width = end - start
                for row in sheet.rows:
                    row += [''] * (sheet.col_count - len(row))
                    row[start:start] = [''] * width
                sheet.formats = {(row, col + width if col >= start else col): fmt
                                 for (row, col), fmt in sheet.formats.items()}
                sheet.col_count += width
            elif 'repeatCell' in request:
                bounds = request['repeatCell']['range']
                sheet = next(s for s in self.sheets.values() if s.id == bounds['sheetId'])
                for row in range(bounds['startRowIndex'], bounds['endRowIndex']):
                    for col in range(bounds['startColumnIndex'], bounds['endColumnIndex']):
                        sheet.formats[row, col] = request['repeatCell']['cell']['userEnteredFormat']
            elif 'appendDimension' in request:
                append = request['appendDimension']
                sheet = next(s for s in self.sheets.values() if s.id == append['sheetId'])
                if append['dimension'] == 'ROWS':
                    sheet.row_count += append['length']
                else:
                    sheet.col_count += append['length']
            else:
                raise AssertionError(f'Unsupported request: {request}')


class Sheet:
    def __init__(self, spreadsheet, title, sheet_id, rows=100, cols=3):
        self.spreadsheet, self.title, self.id = spreadsheet, title, sheet_id
        self.rows = []
        self.formats = {}
        self.row_count, self.col_count = rows, cols
        self.format = Mock()
        self.update_cells = Mock(side_effect=self._update_cells)
        self.append_row = Mock(side_effect=self._append_row)
        self.resize = Mock(side_effect=self._resize)

    def set_cell(self, row, col, value):
        assert row < self.row_count and col < self.col_count
        while len(self.rows) <= row:
            self.rows.append([])
        self.rows[row] += [''] * max(0, col + 1 - len(self.rows[row]))
        self.rows[row][col] = value

    def row_values(self, row, **kwargs):
        return list(self.rows[row - 1]) if row <= len(self.rows) else []

    def col_values(self, col, **kwargs):
        return [row[col - 1] if len(row) >= col else '' for row in self.rows]

    def get_values(self):
        return deepcopy(self.rows)

    def _update_cells(self, cells, **kwargs):
        for cell in cells:
            self.set_cell(cell.row - 1, cell.col - 1, str(cell.value))

    def _append_row(self, values, **kwargs):
        self.rows.append(list(values))
        row = len(self.rows)
        return {'updates': {'updatedRange': f'A{row}:{rowcol_to_a1(row, len(values))}'}}

    def _resize(self, cols):
        self.col_count = cols
