import re
from dataclasses import dataclass, replace
from enum import Enum


class ReviewStage(str, Enum):
    ORAL = "oral"
    WRITTEN = "written"


DEFAULT_REVIEW_STAGES = (ReviewStage.ORAL, ReviewStage.WRITTEN)


class ReviewStatus(str, Enum):
    EMPTY = ""
    SOLVED_WITHOUT_MR = "#"
    READY_TO_BE_CHECKED = "?"
    CHANGES_REQUESTED = "-"
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
        elif oral_marker in ("#", "?", "-", "+") and not written_marker:
            stage, status = ReviewStage.ORAL, ReviewStatus(oral_marker)
        elif written_marker in ("#", "?", "-", "+") and not oral_marker:
            stage, status = ReviewStage.WRITTEN, ReviewStatus(written_marker)
        elif not oral_marker and not written_marker and oral_count == written_count == 0:
            stage, status = ReviewStage.ORAL, ReviewStatus.EMPTY
        else:
            raise ValueError("Conflicting or misplaced review markers")
        if status == ReviewStatus.SOLVED_WITHOUT_MR and (oral_count or written_count):
            raise ValueError("A solution without an MR cannot have review attempts")
        active_count = oral_count if stage == ReviewStage.ORAL else written_count
        if status in (ReviewStatus.READY_TO_BE_CHECKED, ReviewStatus.ACCEPTED) and active_count == 0:
            raise ValueError("A reviewed stage must have at least one attempt")
        return cls(stage, status, oral_count, written_count)


def transition(
    state: ReviewState,
    event: ReviewEvent,
    oral_attempt_limit: int,
    *,
    has_merge_request: bool = False,
    review_stages: tuple[ReviewStage, ...] = DEFAULT_REVIEW_STAGES,
) -> ReviewState:
    if type(oral_attempt_limit) is not int or oral_attempt_limit <= 0:
        raise ValueError("oral_attempt_limit must be a positive integer")
    if not isinstance(event, ReviewEvent):
        raise ValueError("Expected a ReviewEvent")
    if not review_stages or len(set(review_stages)) != len(review_stages):
        raise ValueError("Review stages must be nonempty and unique")
    if any(not isinstance(stage, ReviewStage) for stage in review_stages):
        raise ValueError("Expected ReviewStage values")
    if event != ReviewEvent.TESTS_FAILED and state.status in (
        ReviewStatus.READY_TO_BE_CHECKED, ReviewStatus.CHANGES_REQUESTED,
    ) and state.stage not in review_stages:
        raise ValueError(f"Current review stage {state.stage.value} is disabled for this task")
    match event:
        case ReviewEvent.TESTS_FAILED:
            return state
        case ReviewEvent.TESTS_PASSED:
            return _tests_passed(state, review_stages[0], has_merge_request=has_merge_request)
        case ReviewEvent.ACCEPT | ReviewEvent.CHANGES_ORAL | ReviewEvent.CHANGES_WRITTEN:
            return _manual_review(state, event, oral_attempt_limit, review_stages)
    raise ValueError(f"Unsupported review event: {event!r}")


def _tests_passed(state: ReviewState, first_stage: ReviewStage, *, has_merge_request: bool) -> ReviewState:
    match state.status:
        case ReviewStatus.EMPTY | ReviewStatus.SOLVED_WITHOUT_MR:
            if not has_merge_request:
                return replace(state, stage=first_stage, status=ReviewStatus.SOLVED_WITHOUT_MR)
            return _enter_review(replace(state, stage=first_stage))
        case ReviewStatus.CHANGES_REQUESTED:
            return _enter_review(state)
        case ReviewStatus.READY_TO_BE_CHECKED | ReviewStatus.ACCEPTED | ReviewStatus.FAILED:
            return state
    raise ValueError(f"Unsupported review status: {state.status!r}")


def _enter_review(state: ReviewState) -> ReviewState:
    """Count a new entry into the queue of the explicitly selected stage."""
    match state.stage:
        case ReviewStage.ORAL:
            return replace(state, status=ReviewStatus.READY_TO_BE_CHECKED, oral_attempts=state.oral_attempts + 1)
        case ReviewStage.WRITTEN:
            return replace(state, status=ReviewStatus.READY_TO_BE_CHECKED, written_attempts=state.written_attempts + 1)
    raise ValueError(f"Unsupported review stage: {state.stage!r}")


def _manual_review(
    state: ReviewState, event: ReviewEvent, oral_attempt_limit: int, review_stages: tuple[ReviewStage, ...],
) -> ReviewState:
    if state.status != ReviewStatus.READY_TO_BE_CHECKED:
        raise ValueError("Manual review requires a task waiting for review (?)")
    match event:
        case ReviewEvent.ACCEPT:
            return replace(state, status=ReviewStatus.ACCEPTED)
        case ReviewEvent.CHANGES_ORAL:
            _require_stage(ReviewStage.ORAL, review_stages)
            status = ReviewStatus.CHANGES_REQUESTED
            if state.oral_attempts >= oral_attempt_limit:
                status = ReviewStatus.FAILED
            return replace(state, stage=ReviewStage.ORAL, status=status)
        case ReviewEvent.CHANGES_WRITTEN:
            _require_stage(ReviewStage.WRITTEN, review_stages)
            return replace(state, stage=ReviewStage.WRITTEN, status=ReviewStatus.CHANGES_REQUESTED)
    raise ValueError(f"Unsupported manual review event: {event!r}")


def _require_stage(stage: ReviewStage, review_stages: tuple[ReviewStage, ...]) -> None:
    if stage not in review_stages:
        raise ValueError(f"Review stage {stage.value} is disabled for this task")
