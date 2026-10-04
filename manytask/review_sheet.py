from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import gspread

from .review import MANUAL_REVIEW_EVENTS, ReviewEvent, ReviewStage, ReviewState, ReviewStatus
from .spreadsheet import update_cells_request


REVIEW_DETAILS_SHEET = "review_details"
REVIEW_DETAILS_COLUMNS = (
    "login", "task", "group", "oral_attempts", "last_oral_review_at",
    "codereview_attempts", "last_codereview_at", "first_successful_submission_at",
    "last_successful_submission_at", "stage", "status",
)


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError("Review timestamps must include a timezone")
    return timestamp


def format_timestamp(value: datetime | None) -> str:
    return value.isoformat(sep=" ") if value is not None else ""


@dataclass(frozen=True)
class ReviewTimestamps:
    last_oral_review_at: datetime | None = None
    last_codereview_at: datetime | None = None
    first_successful_submission_at: datetime | None = None
    last_successful_submission_at: datetime | None = None

    def updated(self, event: ReviewEvent, reviewed_stage: ReviewStage, at: datetime) -> "ReviewTimestamps":
        if at.utcoffset() is None:
            raise ValueError("Review timestamps must include a timezone")
        if event == ReviewEvent.TESTS_FAILED:
            return self
        if event == ReviewEvent.TESTS_PASSED:
            return replace(self, first_successful_submission_at=self.first_successful_submission_at or at,
                           last_successful_submission_at=at)
        if event not in MANUAL_REVIEW_EVENTS:
            raise ValueError("Expected a ReviewEvent")
        if reviewed_stage == ReviewStage.ORAL:
            return replace(self, last_oral_review_at=at)
        return replace(self, last_codereview_at=at)


class ReviewDetailsSheet:
    def __init__(self, spreadsheet: gspread.Spreadsheet):
        self.spreadsheet = spreadsheet
        try:
            self.ws = spreadsheet.worksheet(REVIEW_DETAILS_SHEET)
        except gspread.WorksheetNotFound:
            self.ws = spreadsheet.add_worksheet(REVIEW_DETAILS_SHEET, rows=1000, cols=len(REVIEW_DETAILS_COLUMNS))
        header = self.ws.row_values(1)
        legacy_header = tuple(name.replace("codereview_attempts", "written_attempts")
                             .replace("last_codereview_at", "last_written_review_at")
                             for name in REVIEW_DETAILS_COLUMNS)
        if tuple(header) == legacy_header:
            requests = [update_cells_request(self.ws.id, 1, 1, list(REVIEW_DETAILS_COLUMNS))]
            stage_column = REVIEW_DETAILS_COLUMNS.index("stage") + 1
            for row, values in enumerate(self.ws.get_values()[1:], 2):
                if len(values) >= stage_column and values[stage_column - 1] == "written":
                    requests.append(update_cells_request(self.ws.id, row, stage_column, ["codereview"]))
            self.spreadsheet.batch_update({"requests": requests})
            header = list(REVIEW_DETAILS_COLUMNS)
        if not header:
            self.ws.update_cells([gspread.Cell(1, column, name)
                                  for column, name in enumerate(REVIEW_DETAILS_COLUMNS, 1)])
        elif tuple(header) != REVIEW_DETAILS_COLUMNS:
            raise ValueError("Unexpected review_details schema; refusing to overwrite it")

    def _read_timestamps(self, login: str, task: str) -> tuple[int, ReviewTimestamps]:
        rows = self.ws.get_values()
        matches = [(index, row) for index, row in enumerate(rows[1:], 2) if row[:2] == [login, task]]
        if len(matches) > 1:
            raise ValueError(f"Duplicate review_details key: {login}/{task}")
        if not matches:
            return len(rows) + 1, ReviewTimestamps()
        index, row = matches[0]
        values = dict(zip(REVIEW_DETAILS_COLUMNS, row))
        return index, ReviewTimestamps(
            last_oral_review_at=parse_timestamp(values.get("last_oral_review_at", "")),
            last_codereview_at=parse_timestamp(values.get("last_codereview_at", "")),
            first_successful_submission_at=parse_timestamp(values.get("first_successful_submission_at", "")),
            last_successful_submission_at=parse_timestamp(values.get("last_successful_submission_at", "")),
        )

    def write_requests(
        self, row: int, login: str, task: str, group: str, state: ReviewState, dates: ReviewTimestamps,
    ) -> list[dict[str, Any]]:
        show_attempts = state.status not in (ReviewStatus.EMPTY, ReviewStatus.SOLVED_WITHOUT_MR)
        values = [login, task, group, state.oral_attempts if show_attempts else "",
                  format_timestamp(dates.last_oral_review_at),
                  state.codereview_attempts if show_attempts else "", format_timestamp(dates.last_codereview_at),
                  format_timestamp(dates.first_successful_submission_at),
                  format_timestamp(dates.last_successful_submission_at), state.stage.value, state.status.value]
        requests: list[dict[str, Any]] = []
        if row > self.ws.row_count:
            requests.append({"appendDimension": {
                "sheetId": self.ws.id, "dimension": "ROWS", "length": row - self.ws.row_count,
            }})
        requests.append(update_cells_request(self.ws.id, row, 1, values))
        return requests

    def record(
        self, login: str, task: str, group: str, old_state: ReviewState, new_state: ReviewState,
        event: ReviewEvent, at: datetime,
    ) -> None:
        if event == ReviewEvent.TESTS_FAILED:
            return
        row, dates = self._read_timestamps(login, task)
        dates = dates.updated(event, old_state.stage, at)
        self.spreadsheet.batch_update({"requests": self.write_requests(row, login, task, group, new_state, dates)})
