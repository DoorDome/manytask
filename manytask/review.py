import re
from dataclasses import dataclass, replace
from enum import Enum


class ReviewStage(str, Enum):
    ORAL = "oral"
    CODEREVIEW = "codereview"


DEFAULT_REVIEW_STAGES = (ReviewStage.ORAL, ReviewStage.CODEREVIEW)


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
    ACCEPT = "accept"
    REQUEST_ORAL = "request_oral"
    REQUEST_CODEREVIEW = "request_codereview"


MANUAL_REVIEW_EVENTS = (ReviewEvent.ACCEPT, ReviewEvent.REQUEST_ORAL, ReviewEvent.REQUEST_CODEREVIEW)


def parse_manual_review_action(request_type: str) -> ReviewEvent | None:
    for event in MANUAL_REVIEW_EVENTS:
        if event.value == request_type:
            return event
    return None


@dataclass(frozen=True)
class ReviewState:
    stage: ReviewStage = ReviewStage.ORAL
    status: ReviewStatus = ReviewStatus.EMPTY
    oral_attempts: int = 0
    codereview_attempts: int = 0

    def columns(self) -> tuple[str, str]:
        if self.status == ReviewStatus.EMPTY:
            return "", ""
        if self.status == ReviewStatus.SOLVED_WITHOUT_MR:
            return ("#", "") if self.stage == ReviewStage.ORAL else ("", "#")
        oral, codereview = str(self.oral_attempts), str(self.codereview_attempts)
        if self.status == ReviewStatus.FAILED:
            return f"f{oral}", f"f{codereview}"
        if self.status != ReviewStatus.EMPTY:
            if self.stage == ReviewStage.ORAL:
                oral = self.status.value + oral
            else:
                codereview = self.status.value + codereview
        return oral, codereview

    @classmethod
    def from_columns(cls, oral: str, codereview: str) -> "ReviewState":
        def parse(cell: str) -> tuple[str, int]:
            if cell == "":
                return "", 0
            if cell == "#":
                return "#", 0
            match = re.fullmatch(r"([#?+gof-]?)([0-9]+)", cell)
            if match is None:
                raise ValueError(f"Invalid review cell: {cell!r}")
            return match[1], int(match[2])

        oral_marker, oral_count = parse(oral)
        codereview_marker, codereview_count = parse(codereview)
        if (oral_marker, codereview_marker) in (("g", "o"), ("f", "f")):
            stage, status = ReviewStage.ORAL, ReviewStatus.FAILED
        elif oral_marker in ("#", "?", "-", "+") and not codereview_marker:
            stage, status = ReviewStage.ORAL, ReviewStatus(oral_marker)
        elif codereview_marker in ("#", "?", "-", "+") and not oral_marker:
            stage, status = ReviewStage.CODEREVIEW, ReviewStatus(codereview_marker)
        elif not oral_marker and not codereview_marker and oral_count == codereview_count == 0:
            stage, status = ReviewStage.ORAL, ReviewStatus.EMPTY
        else:
            raise ValueError("Conflicting or misplaced review markers")
        if status == ReviewStatus.SOLVED_WITHOUT_MR and (oral_count or codereview_count):
            raise ValueError("A solution without an MR cannot have review attempts")
        active_count = oral_count if stage == ReviewStage.ORAL else codereview_count
        if status in (ReviewStatus.READY_TO_BE_CHECKED, ReviewStatus.ACCEPTED) and active_count == 0:
            raise ValueError("A reviewed stage must have at least one attempt")
        return cls(stage, status, oral_count, codereview_count)


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
        case ReviewEvent.ACCEPT | ReviewEvent.REQUEST_ORAL | ReviewEvent.REQUEST_CODEREVIEW:
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
    """Count oral queue entries; code review requests already count their iteration."""
    match state.stage:
        case ReviewStage.ORAL:
            return replace(state, status=ReviewStatus.READY_TO_BE_CHECKED, oral_attempts=state.oral_attempts + 1)
        case ReviewStage.CODEREVIEW:
            attempts = (max(1, state.codereview_attempts) if state.status == ReviewStatus.CHANGES_REQUESTED
                        else state.codereview_attempts + 1)
            return replace(state, status=ReviewStatus.READY_TO_BE_CHECKED, codereview_attempts=attempts)
    raise ValueError(f"Unsupported review stage: {state.stage!r}")


def _manual_review(
    state: ReviewState, event: ReviewEvent, oral_attempt_limit: int, review_stages: tuple[ReviewStage, ...],
) -> ReviewState:
    if state.status != ReviewStatus.READY_TO_BE_CHECKED:
        raise ValueError("Manual review requires a task waiting for review (?)")
    match event:
        case ReviewEvent.ACCEPT:
            return replace(state, status=ReviewStatus.ACCEPTED)
        case ReviewEvent.REQUEST_ORAL:
            _require_stage(ReviewStage.ORAL, review_stages)
            status = ReviewStatus.CHANGES_REQUESTED
            if state.oral_attempts >= oral_attempt_limit:
                status = ReviewStatus.FAILED
            return replace(state, stage=ReviewStage.ORAL, status=status)
        case ReviewEvent.REQUEST_CODEREVIEW:
            _require_stage(ReviewStage.CODEREVIEW, review_stages)
            return replace(state, stage=ReviewStage.CODEREVIEW, status=ReviewStatus.CHANGES_REQUESTED,
                           codereview_attempts=state.codereview_attempts + 1)
    raise ValueError(f"Unsupported manual review event: {event!r}")


def _require_stage(stage: ReviewStage, review_stages: tuple[ReviewStage, ...]) -> None:
    if stage not in review_stages:
        raise ValueError(f"Review stage {stage.value} is disabled for this task")
