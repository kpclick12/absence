# Modeling Principles

This repository follows a deliberate feature policy for phase 1.

## Main Principle

Prioritize:

- stable features
- interpretable features
- features that are likely to exist consistently across years

Avoid:

- administrative identifiers that change over time
- features that are hard to trust or explain
- local context features that make otherwise similar students receive different scores only because of school/class structure

## Included Across Models

The three current models primarily use:

- student attendance history
- `giltig` and `ogiltig` patterns
- partial absence behavior
- streak and recency features
- term progression
- stable calendar-gap features such as whether a scored day or target day follows a long school break
- prior-year attendance summaries when available
- schedule structure where relevant

## Excluded Across Models

All three models currently exclude these predictors:

- `school_id`
- `class_id`
- school-level rolling context rates
- class-level rolling context rates

These values may still appear in output tables for filtering, reporting, and class-level scoring views, but they are not used as model inputs.

## Why School/Class Features Are Excluded

The current policy is:

- two students with the same attendance history and the same stable predictors should receive the same risk score even if they belong to different schools or classes

This reduces dependence on:

- renamed classes
- reorganized schools
- transfers between units
- school-specific registration habits

## New Students and Förskoleklass

Students with no prior municipal history are still scored.

The models use:

- current-term behavior
- schedule structure where relevant
- a `has_prior_year_history` indicator

This makes early scores possible without pretending prior history exists when it does not.
