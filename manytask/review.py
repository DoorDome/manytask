import re
from dataclasses import dataclass, replace
from enum import Enum


class ReviewStage(str, Enum):
    ORAL = "oral"
    WRITTEN = "written"


class ReviewStatus(str, Enum):
    EMPTY = ""
    SOLVED_WITHOUT_MR = "#"
    WAITING = "?"
    CHANGES = "-"
    ACCEPTED = "+"
    FAILED = "failed"


class ReviewEvent(str, Enum):
    TESTS_PASSED = "tests_passed"
    TESTS_FAILED = "tests_failed"
    ACCEPT = "approve"
    CHANGES_ORAL = "changes_oral"
    CHANGES_WRITTEN = "changes_written"


MANUAL_REVIEW_EVENTS = (ReviewEvent.ACCEPT, ReviewEvent.CHANGES_ORAL, ReviewEvent.CHANGES_WRITTEN)


@dataclass(frozen=True)
class ReviewState:
    stage: ReviewStage = ReviewStage.ORAL
    status: ReviewStatus = ReviewStatus.EMPTY
    oral_attempts: int = 0
    written_attempts: int = 0

    def columns(self) -> tuple[str, str]:
        oral, written = str(self.oral_attempts), str(self.written_attempts)
        if self.status == ReviewStatus.FAILED:
            return f"g{oral}", f"o{written}"
        if self.status == ReviewStatus.ACCEPTED:
            return oral, f"+{written}"
        if self.status != ReviewStatus.EMPTY:
            if self.stage == ReviewStage.ORAL:
                oral = self.status.value + oral
            else:
                written = self.status.value + written
        return oral, written

    @classmethod
    def from_columns(cls, oral: str, written: str) -> "ReviewState":
        def parse(cell: str) -> tuple[str, int]:
            if cell == "":
                return "", 0
            match = re.fullmatch(r"([#?+go-]?)([0-9]+)", cell)
            if match is None:
                raise ValueError(f"Invalid review cell: {cell!r}")
            return match[1], int(match[2])

        oral_marker, oral_count = parse(oral)
        written_marker, written_count = parse(written)
        if (oral_marker, written_marker) == ("g", "o"):
            stage, status = ReviewStage.ORAL, ReviewStatus.FAILED
        elif oral_marker in ("#", "?", "-") and not written_marker:
            stage, status = ReviewStage.ORAL, ReviewStatus(oral_marker)
        elif written_marker in ("?", "-", "+") and not oral_marker:
            stage, status = ReviewStage.WRITTEN, ReviewStatus(written_marker)
        elif not oral_marker and not written_marker and oral_count == written_count == 0:
            stage, status = ReviewStage.ORAL, ReviewStatus.EMPTY
        else:
            raise ValueError("Conflicting or misplaced review markers")
        if status == ReviewStatus.SOLVED_WITHOUT_MR and (oral_count or written_count):
            raise ValueError("A solution without an MR cannot have review attempts")
        active_count = oral_count if stage == ReviewStage.ORAL else written_count
        if status in (ReviewStatus.WAITING, ReviewStatus.ACCEPTED) and active_count == 0:
            raise ValueError("A reviewed stage must have at least one attempt")
        return cls(stage, status, oral_count, written_count)


def transition(
    state: ReviewState,
    event: ReviewEvent,
    oral_attempt_limit: int,
    *,
    has_merge_request: bool = False,
) -> ReviewState:
    if type(oral_attempt_limit) is not int or oral_attempt_limit <= 0:
        raise ValueError("oral_attempt_limit must be a positive integer")
    if not isinstance(event, ReviewEvent):
        raise ValueError("Expected a ReviewEvent")
    if event == ReviewEvent.TESTS_FAILED:
        return state
    if event == ReviewEvent.TESTS_PASSED:
        first_review = state.status in (ReviewStatus.EMPTY, ReviewStatus.SOLVED_WITHOUT_MR)
        if first_review and not has_merge_request:
            return replace(state, status=ReviewStatus.SOLVED_WITHOUT_MR)
        if not first_review and state.status != ReviewStatus.CHANGES:
            return state
        stage = ReviewStage.ORAL if first_review else state.stage
        if stage == ReviewStage.ORAL:
            if state.oral_attempts >= oral_attempt_limit:
                return replace(state, stage=stage, status=ReviewStatus.FAILED)
            return replace(state, stage=stage, status=ReviewStatus.WAITING, oral_attempts=state.oral_attempts + 1)
        return replace(state, status=ReviewStatus.WAITING, written_attempts=state.written_attempts + 1)

    if state.status != ReviewStatus.WAITING:
        raise ValueError("Manual review requires a task waiting for review (?)")
    if event == ReviewEvent.ACCEPT:
        if state.stage != ReviewStage.WRITTEN:
            raise ValueError("Written review is mandatory before acceptance")
        return replace(state, status=ReviewStatus.ACCEPTED)
    stage = ReviewStage.ORAL if event == ReviewEvent.CHANGES_ORAL else ReviewStage.WRITTEN
    status = ReviewStatus.CHANGES
    if stage == ReviewStage.ORAL and state.oral_attempts >= oral_attempt_limit:
        status = ReviewStatus.FAILED
    return replace(state, stage=stage, status=status)
