from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import Any, Callable, Iterable

import gspread
from cachelib import BaseCache
from gspread.utils import ValueInputOption, ValueRenderOption, a1_to_rowcol, absolute_range_name

from .config import ManytaskConfig, ManytaskDeadlinesConfig
from .course import get_current_time
from .glab import Student
from .review import DEFAULT_REVIEW_STAGES, ReviewEvent, ReviewStage, ReviewState, ReviewStatus, transition
from .review_sheet import ReviewDetailsSheet
from .spreadsheet import update_cells_request


logger = logging.getLogger(__name__)


GROUP_ROW_FORMATTING = {
    "backgroundColor": {
        "red": 182.0 / 255.0,
        "green": 215.0 / 255.0,
        "blue": 168.0 / 255.0,
    },
    "borders": {
        "bottom": {
            "style": "SOLID",
        },
    },
    "textFormat": {
        "fontFamily": "Amatic SC",
        "fontSize": 24,
        "bold": True,
    },
}

HEADER_ROW_FORMATTING = {
    "backgroundColor": {
        "red": 217.0 / 255.0,
        "green": 234.0 / 255.0,
        "blue": 211.0 / 255.0,
    },
    "borders": {
        "bottom": {
            "style": "SOLID",
        },
    },
    "textFormat": {
        "fontFamily": "Comfortaa",
        "fontSize": 10,
        "bold": True,
    },
}


# NB: numeration start with 1
@dataclass
class PublicAccountsSheetOptions:
    GROUPS_ROW: int = 1
    MAX_SCORES_ROW: int = 2
    HEADER_ROW: int = 3
    SUBHEADER_ROW: int = 4
    STUDENTS_START_ROW: int = 5

    GITLAB_COLUMN: int = 1
    LOGIN_COLUMN: int = 2
    NAME_COLUMN: int = 3
    TASK_SCORES_START_COLUMN: int = 4

    COLUMNS_PER_TASK: int = 4
    ORAL_OFFSET: int = 1
    WRITTEN_OFFSET: int = 2
    REVIEWER_OFFSET: int = 3


class LoginNotFound(KeyError):
    pass


class TaskNotFound(KeyError):
    pass


class RatingTable:
    class SubmissionStatus:
        def __init__(self, score: int, review: str, reviewer: str | None):
            self.score = score
            self.review = review
            self.reviewer = reviewer

    def __init__(
        self,
        worksheet: gspread.Worksheet,
        cache: BaseCache,
    ):
        self._cache = cache
        self.ws = worksheet

    def get_scores(
        self,
        username: str,
    ) -> dict[str, int]:
        scores = self._cache.get(f"{self.ws.id}:scores:{username}")
        if scores is None:
            scores = {}
        # logger.info(f"scores for {username}: {scores}")
        return scores

    def update_reviewers_list(
        self,
        reviewers: list[str],
        check_order: bool = True,
    ) -> None:
        updated_reviewers = []
        if check_order:
            current_reviewers = self._get_reviewers_queue()
            preserved_reviewers = [name for name in current_reviewers if name in reviewers]
            new_reviewers = [name for name in reviewers if not (name in current_reviewers)]
            updated_reviewers = new_reviewers + preserved_reviewers
        else:
            updated_reviewers = reviewers
        self._cache.set(f"{self.ws.id}:reviewers", updated_reviewers)

    def pop_reviewer(
        self,
    ) -> str | None:
        current_reviewers = self._get_reviewers_queue()
        if not current_reviewers:
            return None
        logger.info(f"Reviewers order = {current_reviewers}")
        reviewer = current_reviewers[0]
        self.update_reviewers_list(current_reviewers[1:] + [reviewer], check_order=False)
        return reviewer

    def _get_reviewers_queue(
        self,
    ) -> list[str]:
        reviewers = self._cache.get(f"{self.ws.id}:reviewers")
        if reviewers is None:
            reviewers = []
        return reviewers

    def update_scores(
        self,
        username: str,
        scores_data: dict[str, int],
    ) -> None:
        self._cache.set(f"{self.ws.id}:scores:{username}", scores_data)

    def get_reviews(
        self,
        username: str,
    ) -> dict[str, ReviewStatus]:
        reviews = self._cache.get(f"{self.ws.id}:reviews:{username}")
        if reviews is None:
            reviews = {}
        return reviews

    def update_reviews(
        self,
        username: str,
        reviews_data: dict[str, ReviewStatus],
    ) -> None:
        self._cache.set(f"{self.ws.id}:reviews:{username}", reviews_data)

    def get_bonus_score(
        self,
        username: str,
    ) -> int:
        bonus_scores = self._cache.get(f"{self.ws.id}:bonus")
        if bonus_scores is None:
            return 0
        return bonus_scores.get(username, 0)

    def get_all_scores_reviews(self) -> dict[str, dict[str, tuple[int, ReviewStatus, str | None]]]:
        all_scores = self._cache.get(f"{self.ws.id}:scores_reviews")
        if all_scores is None:
            all_scores = {}
        return all_scores

    def get_stats(self) -> dict[str, float]:
        stats = self._cache.get(f"{self.ws.id}:stats")
        if stats is None:
            stats = {}
        return stats

    def get_scores_update_timestamp(self) -> str:
        timestamp = self._cache.get(f"{self.ws.id}:update-timestamp")
        if timestamp is None:
            timestamp = "None"
        return timestamp

    def _gather_worksheet_data(self) -> list:
        raw_values = self.ws.get_values()
        # logger.info(f"raw_values: {raw_values}")
        # logger.info(f"raw_values len: {len(raw_values)}")
        if len(raw_values) < PublicAccountsSheetOptions.STUDENTS_START_ROW:
            return list()

        result = list()
        header = raw_values[PublicAccountsSheetOptions.HEADER_ROW - 1]
        # logger.info(f"header: {header}")

        for row in raw_values[PublicAccountsSheetOptions.STUDENTS_START_ROW - 1:]:
            user_data = {"params": dict(), "tasks": dict()}
            # logger.info(f"user: {row[PublicAccountsSheetOptions.LOGIN_COLUMN - 1]}")
            for index, value in enumerate(row[:PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1]):
                user_data["params"][header[index]] = value
            for index in range(PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1, min(len(row), len(header)), PublicAccountsSheetOptions.COLUMNS_PER_TASK):
                if not header[index]:
                    continue
                user_data["tasks"][header[index]] = tuple(row[index:index + PublicAccountsSheetOptions.COLUMNS_PER_TASK])
            result.append(user_data)
        return result

    def update_cached_scores(self) -> None:
        _current_timestamp = get_current_time()

        processed_data = self._gather_worksheet_data()
        # logger.info(f"processed_data: {processed_data}")

        all_scores_and_reviews = {
            user_data["params"]["login"]: {
                k: (
                    int(v[0]),
                    ReviewState.from_columns(v[1] if len(v) > 1 else "", v[2] if len(v) > 2 else "").status,
                    v[3] if len(v) > 3 and v[3] else None,
                )
                for k, v in user_data["tasks"].items()
                if len(v[0]) > 0
            }
            for user_data in processed_data
        }
        # logger.info(f"all_scores_and_reviews: {all_scores_and_reviews}")

        users_score_cache = {}
        for username, user_data in all_scores_and_reviews.items():
            scores = {}
            reviews = {}
            for task, data in user_data.items():
                scores[task] = data[0]
                reviews[task] = data[1]
            users_score_cache[f"{self.ws.id}:scores:{username}"] = scores
            users_score_cache[f"{self.ws.id}:reviews:{username}"] = reviews

        # logger.info(f"{users_score_cache}: users_score_cache")
        all_users_bonus_scores = {
            user_data["params"]["login"]: int(user_data["params"].get("bonus", "")) if user_data["params"].get("bonus", "") else 0
            for user_data in processed_data
        }

        # clear cache saving config
        _config = self._cache.get("__config__")
        config = ManytaskConfig(**_config)

        # get all tasks stats
        _tasks_stats: defaultdict[str, int] = defaultdict(int)
        for tasks in all_scores_and_reviews.values():
            for task_name in tasks.keys():
                _tasks_stats[task_name] += 1
        tasks_stats: dict[str, float] = {
            task.name: (_tasks_stats[task.name] / len(all_scores_and_reviews) if len(all_scores_and_reviews) != 0 else 0)
            for task in config.get_tasks(enabled=True, started=True)
        }

        reviewers_order = self._cache.get(f"{self.ws.id}:reviewers")

        self._cache.clear()
        self._cache.set("__config__", _config)
        self._cache.set(f"{self.ws.id}:scores_reviews", all_scores_and_reviews)
        self._cache.set(f"{self.ws.id}:bonus", all_users_bonus_scores)
        self._cache.set(f"{self.ws.id}:stats", tasks_stats)
        self._cache.set(f"{self.ws.id}:update-timestamp", _current_timestamp)
        self._cache.set(f"{self.ws.id}:reviewers", reviewers_order)
        self._cache.set_many(users_score_cache)

    @staticmethod
    def _read_task_values(row_values: list[str], column: int) -> tuple[int | None, ReviewState, str | None]:
        options = PublicAccountsSheetOptions
        cells = row_values[column - 1:column - 1 + options.COLUMNS_PER_TASK]
        cells += [""] * (options.COLUMNS_PER_TASK - len(cells))
        score = int(cells[0]) if cells[0] != "" else None
        state = ReviewState.from_columns(cells[options.ORAL_OFFSET], cells[options.WRITTEN_OFFSET])
        return score, state, cells[options.REVIEWER_OFFSET] or None

    def store_score(
        self,
        student: Student,
        task_name: str,
        update_fn: Callable[..., Any],
        event: ReviewEvent,
        *,
        oral_attempt_limit: int,
        group_name: str,
        at: datetime,
        has_merge_request: bool = False,
        review_stages: tuple[ReviewStage, ...] = DEFAULT_REVIEW_STAGES,
    ) -> SubmissionStatus:
        if at.utcoffset() is None:
            raise ValueError("Review timestamps must include a timezone")
        column = self._find_task_column(task_name)
        try:
            row = self._find_login_row(student.username)
            values = self.ws.row_values(row)
        except LoginNotFound:
            row, values = None, []
        score, old_state, reviewer = self._read_task_values(values, column)
        new_state = transition(
            old_state, event, oral_attempt_limit, has_merge_request=has_merge_request, review_stages=review_stages,
        )
        if event == ReviewEvent.TESTS_FAILED:
            return self.SubmissionStatus(score or 0, old_state.status.value, reviewer)

        if event == ReviewEvent.TESTS_PASSED and score is None:
            score = update_fn("")
        if row is None:
            row = self._add_student_row(student)
        if event == ReviewEvent.TESTS_PASSED and has_merge_request and reviewer is None:
            reviewer = self.pop_reviewer()
        self.ws.spreadsheet.batch_update({"requests": [update_cells_request(
            self.ws.id, row, column, [score or 0, *new_state.columns(), reviewer or ""],
        )]})

        # Preserve the existing full-student cache refresh, including other tasks on a cache miss.
        values = self.ws.row_values(row)
        scores, reviews = {}, {}
        for task_column, task in self._list_tasks(with_index=True):
            if task:
                task_score, state, _ = self._read_task_values(values, task_column)
                if task_score is not None:
                    scores[task], reviews[task] = task_score, state.status
        self.update_scores(student.username, scores)
        self.update_reviews(student.username, reviews)
        try:
            ReviewDetailsSheet(self.ws.spreadsheet).record(
                student.username, task_name, group_name, old_state, new_state, event, at,
            )
        except Exception:
            # The main transition succeeded; a summary error must not invite a retry of it.
            logger.exception("Cannot update review_details for %s/%s", student.username, task_name)
        return self.SubmissionStatus(score or 0, new_state.status.value, reviewer)

    def sync_columns(self, deadlines_config: ManytaskDeadlinesConfig) -> None:
        options = PublicAccountsSheetOptions
        tasks = deadlines_config.get_tasks(enabled=True, started=True)
        configured = [task.name for task in tasks]
        task_groups = {
            task.name: group.name
            for group in deadlines_config.get_groups(enabled=True, started=True)
            for task in group.tasks if task.name in configured
        }
        existing_columns = [(column, name) for column, name in self._list_tasks(with_index=True) if name]
        existing = [name for _, name in existing_columns]
        if len(set(existing)) != len(existing):
            raise ValueError("Duplicate task headers in main sheet")
        if any(name not in configured for name in existing):
            raise ValueError("Removing existing tasks is not supported by sheet synchronization")
        if [name for name in configured if name in existing] != existing:
            raise ValueError("Reordering existing tasks is not supported by sheet synchronization")

        group_row = self.ws.row_values(options.GROUPS_ROW)
        labels = {}
        group = ""
        for index, (column, name) in enumerate(existing_columns):
            if column != options.TASK_SCORES_START_COLUMN + index * options.COLUMNS_PER_TASK:
                raise ValueError("Task blocks must be contiguous")
            label = group_row[column - 1] if column <= len(group_row) else ""
            group = label or group
            if group != task_groups[name]:
                raise ValueError(f"Moving existing task {name} to another group is not supported")
            labels[name] = label

        requests = []
        ordered = list(existing)
        new_tasks = [task for task in tasks if task.name not in existing]
        for task in new_tasks:
            position = configured.index(task.name)
            following = next((name for name in configured[position + 1:] if name in ordered), None)
            insert_at = ordered.index(following) if following is not None else len(ordered)
            column_index = options.TASK_SCORES_START_COLUMN - 1 + insert_at * options.COLUMNS_PER_TASK
            requests.append({"insertDimension": {
                "range": {"sheetId": self.ws.id, "dimension": "COLUMNS",
                          "startIndex": column_index, "endIndex": column_index + options.COLUMNS_PER_TASK},
                "inheritFromBefore": True,
            }})
            ordered.insert(insert_at, task.name)
            labels[task.name] = ""

        for task in new_tasks:
            column = options.TASK_SCORES_START_COLUMN + ordered.index(task.name) * options.COLUMNS_PER_TASK
            requests.extend([
                update_cells_request(self.ws.id, options.MAX_SCORES_ROW, column, [task.score]),
                update_cells_request(self.ws.id, options.HEADER_ROW, column, [task.name]),
                update_cells_request(
                    self.ws.id, options.SUBHEADER_ROW, column, ["score", "oral", "written", "reviewer"],
                ),
            ])
            for start_row, end_row, formatting in (
                (options.GROUPS_ROW - 1, options.GROUPS_ROW, GROUP_ROW_FORMATTING),
                (options.MAX_SCORES_ROW - 1, options.SUBHEADER_ROW, HEADER_ROW_FORMATTING),
            ):
                requests.append({"repeatCell": {
                    "range": {"sheetId": self.ws.id, "startRowIndex": start_row, "endRowIndex": end_row,
                              "startColumnIndex": column - 1,
                              "endColumnIndex": column - 1 + options.COLUMNS_PER_TASK},
                    "cell": {"userEnteredFormat": formatting}, "fields": "userEnteredFormat",
                }})

        hidden_columns = self._hidden_columns() if existing_columns else set()
        old_columns = {name: column for column, name in existing_columns}
        for index, task in enumerate(tasks):
            column_index = options.TASK_SCORES_START_COLUMN - 1 + index * options.COLUMNS_PER_TASK
            if task.name not in old_columns:
                # Inserted dimensions inherit visibility; reset the entire new block first.
                requests.append(self._column_visibility_request(column_index, options.COLUMNS_PER_TASK, False))
            for stage, offset in ((ReviewStage.ORAL, options.ORAL_OFFSET),
                                  (ReviewStage.WRITTEN, options.WRITTEN_OFFSET)):
                hidden = stage not in task.review_stages
                was_hidden = (old_columns[task.name] - 1 + offset in hidden_columns
                              if task.name in old_columns else False)
                if hidden != was_hidden:
                    requests.append(self._column_visibility_request(column_index + offset, 1, hidden))

        previous_group = None
        for index, name in enumerate(ordered):
            group = task_groups[name]
            label = group if group != previous_group else ""
            if label != labels[name]:
                column = options.TASK_SCORES_START_COLUMN + index * options.COLUMNS_PER_TASK
                requests.append(update_cells_request(self.ws.id, options.GROUPS_ROW, column, [label]))
            previous_group = group
        if requests:
            # Sheets shifts existing cells, formulas and ranges; no old task block is rewritten.
            self.ws.spreadsheet.batch_update({"requests": requests})

    def _hidden_columns(self) -> set[int]:
        metadata = self.ws.spreadsheet.fetch_sheet_metadata(params={
            "ranges": absolute_range_name(self.ws.title),
            "fields": "sheets(properties(sheetId),data(startColumn,columnMetadata(hiddenByUser)))",
        })
        sheet = next(sheet for sheet in metadata["sheets"] if sheet["properties"]["sheetId"] == self.ws.id)
        return {
            block.get("startColumn", 0) + offset
            for block in sheet.get("data", [])
            for offset, column in enumerate(block.get("columnMetadata", []))
            if column.get("hiddenByUser", False)
        }

    def _column_visibility_request(self, start: int, width: int, hidden: bool) -> dict[str, Any]:
        return {"updateDimensionProperties": {
            "range": {"sheetId": self.ws.id, "dimension": "COLUMNS", "startIndex": start, "endIndex": start + width},
            "properties": {"hiddenByUser": hidden}, "fields": "hiddenByUser",
        }}

    def _get_row_values(
        self,
        row: int,
        start: int | None = None,
        step: int | None = None,
        with_index: bool = False,
    ) -> Iterable[Any]:
        values: Iterable[Any] = self.ws.row_values(row, value_render_option=ValueRenderOption.unformatted)
        if with_index:
            values = enumerate(values, start=1)
        if start:
            step = step if step else 1
            values = islice(values, start, None, step)
        return values

    def _list_tasks(
        self,
        with_index: bool = False,
    ) -> Iterable[Any]:
        return self._get_row_values(
            PublicAccountsSheetOptions.HEADER_ROW,
            start=PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1,
            step=PublicAccountsSheetOptions.COLUMNS_PER_TASK,
            with_index=with_index,
        )

    def _find_task_column(
        self,
        task: str,
    ) -> int:
        logger.info(f'Looking for task "{task}"...')
        logger.info(list(self._list_tasks()))
        logger.info(str(task))
        for col, found_task in self._list_tasks(with_index=True):
            if task == found_task:
                return col
        raise TaskNotFound(f'Task "{task}" not found in spreadsheet')

    def _find_login_row(
        self,
        login: str,
    ) -> int:
        logger.info(f'Looking for student "{login}"...')
        all_logins = self.ws.col_values(
            PublicAccountsSheetOptions.LOGIN_COLUMN,
            value_render_option=ValueRenderOption.unformatted,
        )

        for row, found_login in islice(enumerate(all_logins, start=1), PublicAccountsSheetOptions.HEADER_ROW, None):
            if str(found_login) == login:
                return row

        raise LoginNotFound(f"Login {login} not found in spreadsheet")

    def _add_student_row(
        self,
        student: Student,
    ) -> int:
        logger.info(f'Adding student "{student.username}" with name "{student.name}"...')
        if len(student.name) == 0 or re.match(r"\W", student.name, flags=re.UNICODE):
            raise ValueError(f'Name "{student.name}" looks fishy')

        TASKS_RANGE: str = f'INDIRECT(ADDRESS(ROW(), {PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN}) & ":" & ROW())'

        column_to_values_dict = {
            PublicAccountsSheetOptions.GITLAB_COLUMN: self.create_student_repo_link(student),
            PublicAccountsSheetOptions.LOGIN_COLUMN: student.username,
            PublicAccountsSheetOptions.NAME_COLUMN: student.name,
        }

        # fill empty columns with empty string
        row_values = [column_to_values_dict.get(i + 1, "") for i in range(max(column_to_values_dict.keys()))]

        result = self.ws.append_row(
            values=row_values,
            value_input_option=ValueInputOption.user_entered,  # don't escape link
            # note logical table to upend to (gdoc implicit split it to logical tables)
            table_range=f"A{PublicAccountsSheetOptions.STUDENTS_START_ROW}",
        )

        updated_range = result["updates"]["updatedRange"]
        updated_range_upper_bound = updated_range.split(":")[1]
        row_count, _ = a1_to_rowcol(updated_range_upper_bound)
        return row_count

    @staticmethod
    def create_student_repo_link(
        student: Student,
    ) -> str:
        return f'=HYPERLINK("{student.repo}";"git")'
