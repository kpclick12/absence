# Chronic Risk Model

This document describes the current phase-1 chronic absence model implemented in this repository.

## Purpose

The chronic model estimates, for each student on each scored school day, the probability that the student will finish the current `term` above a missed-attendance threshold.

Two outputs are produced:

- `risk_20pct_term_end`: probability of ending the current term at `>= 20%` missed instructional minutes
- `risk_10pct_term_end`: probability of ending the current term at `>= 10%` missed instructional minutes

The model is designed for Swedish school terms:

- `HT`: August-December
- `VT`: January-June

It does **not** currently predict whole-`läsår` end risk.

## Target Definition

The model scores one `student-term-day` at a time.

For a scoring date inside a term, the target is:

- whether the student's final `term-end` missed instructional minutes
- divided by the student's final `term-end` recorded instructional minutes
- is at or above the selected threshold (`10%` or `20%`)

Important implementation detail:

- the current pipeline uses `recorded` instructional minutes, not all theoretically scheduled minutes
- lessons with missing attendance registration are excluded from model labels and denominators

## Training and Prediction Cycle

Recommended operational cycle:

1. Retrain before a new term starts.
2. During the active term, score students daily.
3. Use the score as an early-warning probability, not a hard decision.

The preferred data split is:

- train on prior years/terms
- validate on a later prior year
- keep the newest year untouched as the final out-of-time test

Example:

- train: `2022/23`, `2023/24`, `2024/25`
- final test: `2025/26`

## Inputs Required

The chronic model depends on these tables:

- `attendance_events`
- `student_schedule`
- `school_calendar`

Required columns are enforced in [contracts.py](/Users/johanhellenas/Desktop/projects_codex/absence/src/absence_modeling/contracts.py).

## What the Model Uses

The chronic model is intentionally a `student trajectory` model.

It uses:

- age/stage and term position
- current-day missed minutes and ratios
- rolling recent attendance history
- separate `giltig` and `ogiltig` patterns
- partial absence behavior
- streak and recency features
- current term cumulative ratios
- prior-year attendance summary, when available

The exact feature lists are defined in [tasks.py](/Users/johanhellenas/Desktop/projects_codex/absence/src/absence_modeling/tasks.py), mainly via:

- `CHRONIC_CONTEXT_FREE_NUMERIC_FEATURES`
- `CHRONIC_CONTEXT_FREE_CATEGORICAL_FEATURES`

### Feature Groups

#### 1. Age, stage, and term position

- `grade_numeric`
- `stage`
- `school_day_index`
- `term_progress_ratio`
- `term_days_remaining`
- `weekday_name`

Why:

- age patterns differ across `F-3`, `4-6`, and `7-9`
- the same current absence level means different things early vs late in the term

#### 2. Current-day attendance state

- `daily_recorded_lessons`
- `daily_recorded_minutes`
- `daily_missed_minutes`
- `daily_missed_ratio`
- `daily_giltig_ratio`
- `daily_ogiltig_ratio`
- `daily_partial_lesson_ratio`
- `daily_substantial_miss_lessons`

Why:

- describes the most recent observed state of attendance
- keeps `giltig`, `ogiltig`, and partial absence separate

#### 3. Rolling recent history

For windows `1`, `3`, `5`, `10`, and `20` school days:

- missed ratio
- `giltig` ratio
- `ogiltig` ratio
- partial absence ratio
- substantial-miss day rate

Why:

- chronic risk is driven by persistence and change over time
- short windows capture recent deterioration
- longer windows capture sustained patterns

#### 4. Streak and recency

- `substantial_streak_days`
- `ogiltig_streak_days`
- `school_days_since_substantial_miss`
- `school_days_since_ogiltig_miss`

Why:

- repeated consecutive misses are strong warning signals
- recency helps distinguish active escalation from older history

#### 5. Current term accumulation

- `term_cumulative_missed_ratio`
- `term_cumulative_giltig_ratio`
- `term_cumulative_ogiltig_ratio`
- `trend_5_vs_20_missed_ratio`
- `gap_to_10pct`
- `gap_to_20pct`

Why:

- these features connect the student's current trajectory to the end-of-term threshold being predicted

#### 6. Prior-year history

- `has_prior_year_history`
- `prior_year_missed_ratio`
- `prior_year_giltig_ratio`
- `prior_year_ogiltig_ratio`
- `prior_year_substantial_day_rate`

Why:

- improves early-term scoring when current-term evidence is still limited

## What the Model Does Not Use

The chronic model currently excludes:

- `subject`
- `school_id`
- `class_id`
- school-level and class-level rolling context rates
- demographics
- intervention/support data
- teacher identifiers

This is intentional. The current design principle is:

- two students with the same attendance history and the same individual predictors should receive the same chronic-risk score even if they are in different schools or classes

## New Students and Förskoleklass

Students in `förskoleklass`, or any student new to the municipality, may have no prior-year history.

In that case:

- `has_prior_year_history = 0`
- prior-year ratio features are missing
- the model falls back to current-term behavior only

This means:

- the student can still be scored from the start of the term
- early-term predictions are less informed than for students with prior history

## Model Type

Phase 1 uses:

- `regularized logistic regression` as the default model
- `Explainable Boosting Machine` as an optional challenger if installed

Final probabilities are calibrated separately by stage.

## How to Interpret the Score

The score is:

- a probability estimate
- not a diagnosis
- not a causal explanation

Example:

- `risk_20pct_term_end = 0.34`

means:

- based on the student's current and historical attendance pattern
- the model estimates a `34%` chance that the student will finish the current term at or above `20%` missed instructional minutes

## Known Limitations

- The target is `term-end`, not `läsår-end`.
- Missing attendance registration is excluded rather than modeled.
- If late administrative corrections are common, backtests may still overstate real-world quality unless true as-of snapshots are reconstructed.
- No causal interpretation should be made from the score.
