# Short-Horizon Student Risk Model

This model replaces the earlier `next-day` student model.

## What It Predicts

For each student on each scored school day, the model estimates:

- the probability that the student will have at least one `substantial absence`
- within the next `N school days`

Default:

- `N = 5`

A substantial absence means the student misses at least `50%` of a scheduled lesson.

## Why This Target Exists

This model is meant to support the practical question:

- is this student currently moving into an `absence-mode` pattern?

It is intentionally broader than exact tomorrow prediction, because:

- a lot of single-day absence is random
- a 3-5 day horizon is usually more stable
- staff can still act on a high score quickly

## Training Label

One training row equals:

- one `student`
- on one scored `date`

The label is positive if, in the next `N school days`, the student has at least one lesson with:

- `missed_minutes / scheduled_minutes >= 0.50`

Training rows are only kept when the full future horizon is observable within the same academic year and all future days in that horizon have complete attendance registration.

## Core Feature Groups

The model uses:

- recent daily missed-minute ratios over `1, 3, 5, 10, 20` school-day windows
- recent `giltig` and `ogiltig` ratios
- partial absence patterns
- substantial-miss and `ogiltig` streaks
- days since the last substantial miss / `ogiltig` miss
- current term cumulative missed-minute ratios
- prior-year attendance summaries when available
- current day calendar-gap features
- upcoming short-horizon schedule load
- next school day schedule shape

## Upcoming Schedule Features

The short-horizon schedule features are intentionally simple and stable:

- `future_school_days_available`
- `future_total_scheduled_lessons`
- `future_total_scheduled_minutes`
- `future_average_daily_minutes`
- `future_total_subject_count`
- `future_morning_lessons`
- `future_afternoon_lessons`
- `next_school_day_total_scheduled_lessons`
- `next_school_day_total_scheduled_minutes`
- `next_school_day_first_lesson_start`
- `next_school_day_last_lesson_start`
- `next_school_day_subject_count`
- `next_school_day_morning_lessons`
- `next_school_day_afternoon_lessons`
- `next_school_day_gap_days`
- `next_school_day_after_long_break`
- `next_school_day_weekday_name`

## Explicitly Excluded

The model does not use:

- `school_id`
- `class_id`
- school/class rolling context rates
- teacher identifiers
- demographics
- intervention logs

These values may still appear in outputs for filtering.

## New Students and Förskoleklass

Students without prior municipal history are still scored.

The model uses:

- current-term attendance pattern
- current short-horizon schedule pattern
- `has_prior_year_history`

So early scores are still possible without pretending prior history exists.

## Output Interpretation

The output is a calibrated probability.

Example:

- `0.34` means the model estimates a `34%` chance that the student will have at least one substantial absence within the next `5 school days`

It should be used as:

- a ranked early-warning signal

not as:

- a deterministic forecast
