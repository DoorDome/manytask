# Configurable task review

The review model in `manytask/review.py` is independent of Sheets and GitLab.
`ReviewEvent.TESTS_PASSED` and `TESTS_FAILED` describe automatic results; only
`approve`, `changes_oral` and `changes_written` are manual decisions.

Passing tests first records solved-without-MR (`#`) unless the report supplies
an MR. The first passing report with an MR starts the task's configured first stage. A request for
changes selects the next stage; a later passing report enters that stage's
review queue. Each entry into `?` increments only that stage's attempt counter.
Repeated passing reports while waiting do not create attempts. Approval at either
enabled stage accepts the task, including the first review. Acceptance and
oral-limit failure are terminal.

`deadlines.oral_attempt_limit` defaults to 3 and requires a positive integer.
Requesting another oral round after that many attempts fails permanently;
written review, if enabled, is still allowed after the last permitted oral attempt.
The limit is checked when requesting another oral round; queue entry does not
recheck the limit if course settings changed after that decision.

## Per-task pipeline

Set `review_stages` on each task in the course YAML:

```yaml
tasks:
  - task: oral_only
    score: 10
    review_stages: [oral]
  - task: written_only
    score: 10
    review_stages: [written]
  - task: either_start_oral
    score: 10
    review_stages: [oral, written]
  - task: either_start_written
    score: 10
    review_stages: [written, oral]
```

The list must contain one or two distinct stages. Its first element selects the
initial stage, not a mandatory sequence. One review is active at a time;
`changes_oral` and `changes_written` select the next stage after corrections.
Selecting a disabled stage returns HTTP 409 before sheet writes. `approve`
accepts a task on either enabled stage; passing both stages is not required.
Omitting the field enables both stages and starts with oral review. Unlike the
previous behavior, this default also permits approval at the first oral review.
The MR requirement applies to all pipelines, including oral-only tasks.

Choose stages before starting reviews. Changing the first stage does not move
an active review. If an active stage is disabled, successful submissions and
manual decisions return 409 until the stage is re-enabled or the main-sheet
state is explicitly resolved. Terminal results and historical counts are kept;
no automatic migration or re-opening is performed.

## Transition rules in code

`ReviewStatus.READY_TO_BE_CHECKED` (`?`) means that the selected stage is ready
for a reviewer; `CHANGES_REQUESTED` (`-`) means that corrections are required
before the next review. The stored markers remain unchanged.

`transition` validates the input and dispatches by `ReviewEvent`. `_tests_passed`
selects the next step by status, `_enter_review` handles each stage explicitly
and increments its counter, and `_manual_review` applies the selected decision.
Failed tests, repeated passing reports in the queue and terminal states preserve
the state. Manual decisions require `READY_TO_BE_CHECKED` and never increment
attempt counters.

## Application integration

The main sheet is the source of review state. Each task occupies four columns:
score, oral, written, reviewer. `#0` in the starting stage means solved without
MR (`#0 / 0` for oral, `0 / #0` for written); `?N` marks the
stage waiting for review, and `-N` the next stage after corrections. Inactive
stages retain counts. Acceptance appears as `+N` in the stage that accepted; terminal
failure appears as `gN / oN`. Empty new cells mean no review yet. New courses use
this schema directly; no legacy sheet or cached-object migration is provided.

The API and web cache share `ReviewStatus`. Every report reads the main sheet
before deciding a transition. Rendering and cache refresh never create attempts.
A saved score, including zero, is retained; manual decisions never regrade.

The checker continues to POST `/api/report` with the existing authentication and
fields. A score at least the full task score, before deadline penalties, means
passed tests; omitted score retains the full-score convention. A nonempty,
whitespace-trimmed `merge_request_iid` is required only to start the first review.
After review starts, omitting the optional MR ID does not undo that fact. Merely
creating an MR does not notify Manytask: send another successful report with its
ID. MR closure/deletion is not polled.

Manual request types are `approve`, `changes_oral`, `changes_written` and require
`reported_by` to resolve to a reviewer. Missing identity returns 400, insufficient
permissions 403, and an invalid transition 409 without sheet writes. `reject` is
rejected with 400. Internal automatic event names are not exposed as request
actions; the API derives them from scores. Retrying a completed manual action
returns 409 without another transition. Passing reports while already waiting
keep the same attempt count. There is no request-ID deduplication or cross-worker
read/modify/write lock; serialize reports for a student/task in course CI.

## Course GitLab CI

The live CI definition belongs to the external public course repository. Copy
`ci/review.gitlab-ci.yml` there and include it from the course `.gitlab-ci.yml`:

```yaml
include:
  - local: /ci/review.gitlab-ci.yml
```

Remove obsolete approve/reject jobs there. The template provides exactly three
manual jobs: `review-accept`, `review-changes-oral`, `review-changes-written`.
Configure `MANYTASK_URL` and the existing `TESTER_TOKEN`. Task branches are
`submit/<task>`, student projects must be named by username (`CI_PROJECT_NAME`),
and the launching reviewer is `GITLAB_USER_LOGIN`. Jobs POST URL-encoded fields
to `/api/report`; neither score nor a new MR ID is required for manual decisions.
Verify that the course's automatic reporter actually supplies `merge_request_iid`.
Offline tests do not establish that the external CI has been deployed.

## Independent review summary

`review_details` has one row per `(login, task)`, with these columns in order:

```text
login, task, group, oral_attempts, last_oral_review_at,
written_attempts, last_written_review_at, first_successful_submission_at,
last_successful_submission_at, stage, status
```

Stage, status and counters are copies of the completed main-sheet transition.
They are never read to decide the next transition. Only the four dates are read
back, solely to preserve the summary's first/latest timestamps. They are stored
as timezone-aware ISO strings with a space separator; blank means unknown.

Successful reports, including those before MR creation and after terminal review,
set the first successful date once and update the last successful date. The last
date is from the last processed report, not the maximum of all event dates.
Automatic dates use `submit_time` (`%Y-%m-%d %H:%M:%S%z`) or course time when
omitted. Manual decisions use server time and update the date of the stage that
was reviewed, not the next stage selected by the decision.

The main write finishes before any summary read/write. Failed tests and invalid
actions never access the summary. A summary error is logged with login/task and
does not reject or roll back an already completed review. No dates or hidden
metadata columns are added to the main sheet. After a summary failure, subsequent
reports can refresh its status/counters but cannot reconstruct a lost event date.
There is no append-only event log or automatic retry queue. Sheet creation and
sequential keyed upserts are idempotent; duplicate keys or unexpected headers are
reported instead of overwriting history.

## Extending task groups

`sync_columns` inserts a four-column block for each new configured, enabled,
started task, preserving the configured task/group order. Existing values are
shifted by Google Sheets `insertDimension`, along with formulas and ranges.
Only new headers are formatted. Groups use a label above the first task, not
merged cells; inserting a new first task moves that label. Existing students get
empty cells, and new students use the same schema. Repeating sync is a no-op.
The summary sheet is not read or written during synchronization.

Removing, reordering or moving existing tasks between groups is outside this
operation: such configurations fail before any writes. The operation never
rebuilds or shrinks the sheet to match a new configuration.


## Review column visibility

`sync_columns` keeps all four physical columns per task and hides only an unused
review column with Google Sheets `updateDimensionProperties.hiddenByUser`.
Enabling a stage again unhides its column. Existing values, counters, formulas
and column positions are not rewritten by visibility updates. The task's score
and reviewer columns remain available; a new block explicitly resets inherited
visibility before hiding its unused stage. Synchronization reads current column
metadata, so repeating it with unchanged configuration sends no write requests.
Hiding is a Sheets UI setting, not access control; API reads still include the
hidden columns. The web UI already displays a single aggregate review result.

External formulas that detect acceptance only in the written column must also
check the oral column for `+N`. Four-column offsets remain unchanged, but that
old assumption about the acceptance marker no longer holds.

The API supports this visibility operation; local tests cover request structure,
insertion offsets and re-enabling stages. Live Sheets rendering is not tested.
Reference: [Google Sheets dimension properties](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/sheets#DimensionProperties).
