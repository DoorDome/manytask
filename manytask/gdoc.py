from __future__ import annotations

from copy import deepcopy
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable

import gspread
from authlib.integrations.requests_client import AssertionSession
from cachelib import BaseCache
from google.auth.credentials import AnonymousCredentials
from gspread import Cell as GCell
from gspread.utils import ValueInputOption, ValueRenderOption, a1_to_rowcol, rowcol_to_a1

from .config import ManytaskConfig, ManytaskDeadlinesConfig
from .course import get_current_time
from .glab import Student


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

    GITLAB_COLUMN: int = 1
    LOGIN_COLUMN: int = 2
    NAME_COLUMN: int = 3
    FLAGS_COLUMN: int = 4
    BONUS_COLUMN: int = 5
    TOTAL_COLUMN: int = 6
    PERCENTAGE_COLUMN: int = 7
    TOTAL_WITH_REVIEW_COLUMN: int = 8
    PERCENTAGE_WITH_REVIEW_COLUMN: int = 9
    TASK_SCORES_START_COLUMN: int = 14

    COLUMNS_PER_TASK: int = 2


class LoginNotFound(KeyError):
    pass


class TaskNotFound(KeyError):
    pass


class GoogleDocApi:
    def __init__(
        self,
        base_url: str,
        gdoc_credentials: dict[str, Any],
        public_worksheet_id: str,
        public_scoreboard_sheet: int,
        cache: BaseCache,
    ):
        """
        :param base_url:
        :param gdoc_credentials:
        :param public_worksheet_id:
        :param public_scoreboard_sheet:
        :param cache:
        """
        self._url = base_url
        self._gdoc_credentials = gdoc_credentials
        self._public_worksheet_id = public_worksheet_id
        self._public_scoreboard_sheet = public_scoreboard_sheet

        self._assertion_session = self._create_assertion_session()

        self._public_scores_sheet = self._get_sheet(public_worksheet_id, public_scoreboard_sheet)
        self._cache = cache

    def _create_assertion_session(self) -> AssertionSession:
        """Create AssertionSession to auto refresh access to google api"""
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = self._gdoc_credentials

        header = {"alg": "RS256"}
        if key_id := credentials.get("private_key_id", None):
            header["kid"] = key_id

        # Google puts scope in payload
        claims = {"scope": " ".join(scopes)}
        return AssertionSession(
            token_endpoint=credentials["token_uri"],
            issuer=credentials["client_email"],
            subject=None,
            audience=credentials["token_uri"],
            grant_type=AssertionSession.JWT_BEARER_GRANT_TYPE,
            scope=" ".join(scopes),
            claims=claims,
            key=credentials["private_key"],
            header=header,
        )

    def _get_sheet(
        self,
        worksheet_id: str,
        sheet_id: int,
    ) -> gspread.Worksheet:
        gs: gspread.Client = gspread.Client(AnonymousCredentials(), session=self._assertion_session)
        worksheet: gspread.Spreadsheet = gs.open_by_key(worksheet_id)
        return worksheet.get_worksheet(sheet_id)

    def fetch_rating_table(self) -> "RatingTable":
        return RatingTable(self._public_scores_sheet, self._cache)

    def get_spreadsheet_url(self) -> str:
        return f"{self._url}/spreadsheets/d/{self._public_worksheet_id}#gid={self._public_scoreboard_sheet}"


class RatingTable:
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
    
    def update_scores(
        self,
        username: str,
        scores_data: dict[str, int],
    ) -> None:
        self._cache.set(f"{self.ws.id}:scores:{username}", scores_data)
    
    def get_reviews(
        self,
        username: str,
    ) -> dict[str, str]:
        reviews = self._cache.get(f"{self.ws.id}:reviews:{username}")
        if reviews is None:
            reviews = {}
        return reviews
    
    def update_reviews(
        self,
        username: str,
        reviews_data: dict[str, int],
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

    def get_all_scores_reviews(self) -> dict[str, dict[str, tuple[int, str]]]:
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
        if len(raw_values) <= PublicAccountsSheetOptions.HEADER_ROW:
            return list()

        result = list()
        header = raw_values[PublicAccountsSheetOptions.HEADER_ROW - 1]
        # logger.info(f"header: {header}")

        for row in raw_values[PublicAccountsSheetOptions.HEADER_ROW:]:
            user_data = {"params": dict(), "tasks": dict()}
            # logger.info(f"user: {row[PublicAccountsSheetOptions.LOGIN_COLUMN - 1]}")
            for index, value in enumerate(row[:PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1]):
                user_data["params"][header[index]] = value
            for index in range(PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1, len(row), PublicAccountsSheetOptions.COLUMNS_PER_TASK):
                user_data["tasks"][header[index]] = tuple(row[index:index + PublicAccountsSheetOptions.COLUMNS_PER_TASK])
            result.append(user_data)
        return result

    def update_cached_scores(self) -> None:
        _current_timestamp = get_current_time()

        processed_data = self._gather_worksheet_data()
        # logger.info(f"processed_data: {processed_data}")

        all_scores_and_reviews = {
            user_data["params"]["login"]: {
                k: (int(v[0]),  self._review_string_to_status(v[1] if len(v) > 1 else "")) 
                for k, v in user_data["tasks"].items()
                if len(v[0]) > 0
            }
            for user_data in processed_data
        }
        # logger.info(f"all_scores_and_reviews: {all_scores_and_reviews}")
        
        users_score_cache = {
            f"{self.ws.id}:scores:{username}": {
                task: data[0] 
                for task, data in user_data.items()
            }
            for username, user_data in all_scores_and_reviews.items() 
        } | {
            f"{self.ws.id}:reviews:{username}": {
                task: data[1] 
                for task, data in user_data.items()
            }
            for username, user_data in all_scores_and_reviews.items() 
        }
        # logger.info(f"{users_score_cache}: users_score_cache")
        all_users_bonus_scores = {
            user_data["params"]["login"]: int(user_data["params"]["bonus"]) if user_data["params"]["bonus"] else 0
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

        self._cache.clear()
        self._cache.set("__config__", _config)
        self._cache.set(f"{self.ws.id}:scores_reviews", all_scores_and_reviews)
        self._cache.set(f"{self.ws.id}:bonus", all_users_bonus_scores)
        self._cache.set(f"{self.ws.id}:stats", tasks_stats)
        self._cache.set(f"{self.ws.id}:update-timestamp", _current_timestamp)
        self._cache.set_many(users_score_cache)

    def store_score(
        self,
        student: Student,
        task_name: str,
        update_fn: Callable[..., Any],
        review: bool | None,
    ) -> tuple[int, str]:
        try:
            student_row = self._find_login_row(student.username)
        except LoginNotFound:
            student_row = self._add_student_row(student)

        task_column = self._find_task_column(task_name)

        score_cell = self.ws.cell(student_row, task_column)
        old_score = int(score_cell.value) if score_cell.value else 0

        review_cell = self.ws.cell(student_row, task_column + 1)
        old_review = review_cell.value if review_cell.value else ""

        if review is None:
            flags = self.ws.cell(student_row, PublicAccountsSheetOptions.FLAGS_COLUMN).value
            new_score = update_fn(flags, old_score)
            new_review = old_review
            score_cell.value = new_score
            logger.info(f"Setting score = {new_score}")
        else:
            new_score = old_score

        new_review = self._format_review(old_review, review)
        review_cell.value = new_review
        logger.info(f"Setting review = {new_review}")

        repo_link_cell = GCell(
            student_row,
            PublicAccountsSheetOptions.GITLAB_COLUMN,
            self.create_student_repo_link(student),
        )
        self.ws.update_cells(
            [repo_link_cell, score_cell, review_cell],
            value_input_option=ValueInputOption.user_entered,
        )

        tasks = self._list_tasks(with_index=False)
        scores = self._get_row_values(
            student_row,
            start=PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN - 1,
            step=PublicAccountsSheetOptions.COLUMNS_PER_TASK,
            with_index=False,
        )
        student_scores = {task: score for task, score in zip(deepcopy(tasks), scores) if score or str(score) == "0"}

        reviews = self._get_row_values(
            student_row,
            start=PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN,
            step=PublicAccountsSheetOptions.COLUMNS_PER_TASK,
            with_index=False,
        )
        student_reviews = {task: self._review_string_to_status(review)
                           for task, review in zip(tasks, reviews) 
                           if self._review_string_to_status(review) is not None}

        logger.info(f"Actual scores: {student_scores}")
        logger.info(f"Actual reviews: {student_reviews}")

        self.update_scores(student.username, student_scores)
        self.update_reviews(student.username, student_reviews)
        return new_score, new_review

    def sync_columns(
        self,
        deadlines_config: ManytaskDeadlinesConfig,
    ) -> None:
        max_score = deadlines_config.max_score_started
        groups = deadlines_config.get_groups(enabled=True, started=True)
        tasks = deadlines_config.get_tasks(enabled=True, started=True)
        task_name_to_group_name = {task.name: group.name for group in groups for task in group.tasks if task in tasks}

        # TODO: maintain group orger when adding new task in added group
        logger.info("Syncing rating columns...")
        existing_tasks = list(self._list_tasks(with_index=False))
        existing_task_names = set(task for task in existing_tasks if task)
        tasks_to_create = [task for task in tasks if task.name not in existing_task_names]

        current_worksheet_size = PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN + len(existing_tasks) * PublicAccountsSheetOptions.COLUMNS_PER_TASK - 1
        required_worksheet_size = current_worksheet_size
        if tasks_to_create:
            required_worksheet_size = current_worksheet_size + len(tasks_to_create) * PublicAccountsSheetOptions.COLUMNS_PER_TASK

            self.ws.resize(cols=required_worksheet_size)

            cells_to_update = []
            current_group = None
            for index, task in enumerate(tasks_to_create):
                col = current_worksheet_size + 1 + PublicAccountsSheetOptions.COLUMNS_PER_TASK * index
                cells_to_update.append(GCell(PublicAccountsSheetOptions.HEADER_ROW, col, task.name))
                cells_to_update.append(GCell(PublicAccountsSheetOptions.MAX_SCORES_ROW, col, str(task.score)))

                task_group_name = task_name_to_group_name[task.name]

                if task_group_name != current_group:
                    cells_to_update.append(GCell(PublicAccountsSheetOptions.GROUPS_ROW, col, task_group_name))
                    current_group = task_group_name
        else:
            cells_to_update = []

        if max_score:
            cells_to_update.append(
                GCell(
                    PublicAccountsSheetOptions.GROUPS_ROW,
                    PublicAccountsSheetOptions.TOTAL_COLUMN,
                    str(max_score),
                )
            )

        if cells_to_update:
            self.ws.update_cells(cells_to_update, value_input_option=ValueInputOption.user_entered)

            self.ws.format(
                f"{rowcol_to_a1(PublicAccountsSheetOptions.GROUPS_ROW, PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN)}:"  # noqa: E501
                f"{rowcol_to_a1(PublicAccountsSheetOptions.GROUPS_ROW, required_worksheet_size)}",
                GROUP_ROW_FORMATTING,
            )
            self.ws.format(
                f"{rowcol_to_a1(PublicAccountsSheetOptions.HEADER_ROW, PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN)}:"  # noqa: E501
                f"{rowcol_to_a1(PublicAccountsSheetOptions.HEADER_ROW, required_worksheet_size)}",
                HEADER_ROW_FORMATTING,
            )
            self.ws.format(
                f"{rowcol_to_a1(PublicAccountsSheetOptions.MAX_SCORES_ROW, PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN)}:"  # noqa: E501
                f"{rowcol_to_a1(PublicAccountsSheetOptions.MAX_SCORES_ROW, required_worksheet_size)}",
                HEADER_ROW_FORMATTING,
            )

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
            PublicAccountsSheetOptions.FLAGS_COLUMN: "",
            PublicAccountsSheetOptions.BONUS_COLUMN: "",
            PublicAccountsSheetOptions.TOTAL_COLUMN:
                # total: sum(current row: from RATINGS_COLUMN to inf) + BONUS_COLUMN
                f'=SUM(ARRAYFORMULA(IF(MOD(COLUMN({TASKS_RANGE}), {PublicAccountsSheetOptions.COLUMNS_PER_TASK})=0, '
                f'{TASKS_RANGE}, 0))) '
                f'+ INDIRECT(ADDRESS(ROW(), {PublicAccountsSheetOptions.BONUS_COLUMN}))',
            PublicAccountsSheetOptions.PERCENTAGE_COLUMN:
                # percentage: TOTAL_COLUMN / max_score cell (1st row of TOTAL_COLUMN)
                f"=IFERROR(ROUND(INDIRECT(ADDRESS(ROW(); {PublicAccountsSheetOptions.TOTAL_COLUMN})) "
                f"/ INDIRECT(ADDRESS({PublicAccountsSheetOptions.GROUPS_ROW}; "
                f"{PublicAccountsSheetOptions.TOTAL_COLUMN})), 4); 0) * 100",  # percentage
            PublicAccountsSheetOptions.TOTAL_WITH_REVIEW_COLUMN:
                # total: sum(current row: from RATINGS_COLUMN to inf, multiply by 0 or 1 whether review is passed) + BONUS_COLUMN
                f'=SUM(ARRAYFORMULA(IF(MOD(COLUMN({TASKS_RANGE}), {PublicAccountsSheetOptions.COLUMNS_PER_TASK})=0, '
                f'{TASKS_RANGE} * '
                f'(LEFT(INDIRECT(ADDRESS(ROW(), {PublicAccountsSheetOptions.TASK_SCORES_START_COLUMN + 1}) & ":" & ROW()), 1)="+"), 0))) + '
                f'INDIRECT(ADDRESS(ROW(), {PublicAccountsSheetOptions.BONUS_COLUMN}))',
            PublicAccountsSheetOptions.PERCENTAGE_WITH_REVIEW_COLUMN:
                # percentage: TOTAL_COLUMN / max_score cell (1st row of TOTAL_COLUMN)
                f"=IFERROR(ROUND(INDIRECT(ADDRESS(ROW(); {PublicAccountsSheetOptions.TOTAL_WITH_REVIEW_COLUMN})) "
                f"/ INDIRECT(ADDRESS({PublicAccountsSheetOptions.GROUPS_ROW}; "
                f"{PublicAccountsSheetOptions.TOTAL_COLUMN})), 4); 0) * 100",  # percentage
        }

        # fill empty columns with empty string
        row_values = [column_to_values_dict.get(i + 1, "") for i in range(max(column_to_values_dict.keys()))]

        result = self.ws.append_row(
            values=row_values,
            value_input_option=ValueInputOption.user_entered,  # don't escape link
            # note logical table to upend to (gdoc implicit split it to logical tables)
            table_range=f"A{PublicAccountsSheetOptions.HEADER_ROW + 1}",
        )

        updated_range = result["updates"]["updatedRange"]
        updated_range_upper_bound = updated_range.split(":")[1]
        row_count, _ = a1_to_rowcol(updated_range_upper_bound)
        return row_count
    
    @staticmethod
    def _format_review(old_value: str, review_status: bool | None) -> str:
        # logger.info(f"initial {old_value}, review {review_status}")
        if old_value == "+":
            old_value = "+0"
        elif old_value == "" or old_value == "?":
            old_value = "0"
        elif old_value.startswith("?"):
            old_value = old_value[1:]
        bad_attempts = abs(int(old_value))

        gen_trunkated_string = lambda att: (str(att) if att > 0 else "")

        if review_status is None:
            if old_value.startswith("+"):
                return "'+" + gen_trunkated_string(bad_attempts)
            return "?" + gen_trunkated_string(bad_attempts)
        elif review_status:
            return "'+" + gen_trunkated_string(bad_attempts)
        else:
            return f"'-{bad_attempts + 1}"
        
    @staticmethod
    def _review_string_to_status(review: str) -> bool | None:
        if not review or review.startswith("?"):
            return None
        return review.startswith("+")

    @staticmethod
    def create_student_repo_link(
        student: Student,
    ) -> str:
        return f'=HYPERLINK("{student.repo}";"git")'
