# Two-stage review

The review model in `manytask/review.py` is independent of Sheets and GitLab.
`ReviewEvent.TESTS_PASSED` and `TESTS_FAILED` describe automatic results; only
`approve`, `changes_oral` and `changes_written` are manual decisions.

Passing tests first records solved-without-MR (`#`) unless the report supplies
an MR. The first passing report with an MR starts oral review. A request for
changes selects the next stage; a later passing report enters that stage's
review queue. Each entry into `?` increments only that stage's attempt counter.
Repeated passing reports while waiting do not create attempts. Only written
review can accept a task; acceptance and oral-limit failure are terminal.

`deadlines.oral_attempt_limit` defaults to 3 and requires a positive integer.
Requesting another oral round after that many attempts fails permanently;
written review is still allowed after the last permitted oral attempt.

This commit introduces the pure model and configuration. Application reports
still use the original review path until the table/API integration is installed.
