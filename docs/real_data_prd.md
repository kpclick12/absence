# Scalable Absence Modeling Experiment Platform PRD

## Summary

Build a production-oriented experimentation and deployment workflow for three student absence model families:

- short-horizon risk
- chronic absence risk
- lesson-level risk

The system must support large real-world datasets, avoid running heavy preprocessing inside the LLM loop, and make it possible to iterate on models efficiently while keeping evaluation stable and deployment practical.

Intended use of Codex:

- design and implement the codebase
- propose and edit training logic
- review experiment results from compact summaries

Intended use of the compute and data platform:

- raw data ingestion
- feature materialization
- large-scale model training
- batch scoring
- SQL or in-database deployment where appropriate

## Problem

The current prototype works on synthetic data but will not scale cleanly to real data volumes.

Expected production scale:

- about 50,000 students
- about 4-5 lessons per student per day
- multiple school years
- large lesson-level training sets and expensive feature generation

If every experiment loop iteration reloads raw data and rebuilds all features from scratch, model search will be too slow and expensive. If Codex is used to drive full-data iterative training directly in chat, both runtime and token usage will become impractical.

## Goals

1. Support real-data experimentation for all three model families.
2. Reduce per-experiment runtime by materializing model-ready features once.
3. Keep evaluation stable and time-based.
4. Allow limited agent-driven search over model code without reprocessing raw data every run.
5. Make final models deployable at scale, ideally with SQL scoring for simple model families.
6. Keep production governance separate from experimentation.

## Non-Goals

1. Build a full MLOps platform with monitoring dashboards and approvals UI.
2. Replace the warehouse with Python-only preprocessing.
3. Make the entire training loop SQL-native.
4. Let Codex directly orchestrate large numbers of full-data runs interactively.
5. Support every model family equally from day one if runtime becomes prohibitive.

## Users

- ML engineer
- analytics engineer or data engineer
- product or data science lead
- Codex as implementation and code-editing assistant

## Success Criteria

1. Feature tables for each task can be materialized once and reused across many training runs.
2. A single experiment iteration trains from precomputed feature tables, not raw attendance tables.
3. Short-horizon and chronic loops can run dozens of iterations on sampled or capped training data within practical time.
4. Lesson-level experimentation is possible on sampled or capped data and full-data finalists.
5. Final model artifacts include:
   - code version
   - feature schema
   - metrics
   - model parameters
   - optional SQL export for supported models
6. Promotion to production requires fixed evaluation gates beyond one scalar metric.

## Model Families

### 1. Short-Horizon

Predict near-term absence risk over the configured future school-day horizon.

### 2. Chronic

Predict end-of-term chronic absence risk for:

- 10% threshold
- 20% threshold

Treat chronic as one model family with two targets.

### 3. Lesson

Predict risk of substantial absence at the lesson level.

## Product Requirements

### A. Data Layer

Build warehouse-backed, model-ready tables for each task.

Required feature tables:

- `features_short_horizon`
- `features_chronic_10`
- `features_chronic_20`
- `features_lesson`

Each row must already contain:

- target label
- train, validation, and test split identifiers
- all numeric and categorical model features
- stable entity and date keys
- cohort fields for subgroup analysis

Raw preprocessing must not be rerun inside each experiment iteration.

### B. Split Strategy

Use fixed time-based splits only.

Requirements:

- preserve current academic-year split logic
- define:
  - train split
  - validation split
  - test split
- support multiple historical backtest windows
- prevent leakage from future dates into feature computation

### C. Training Workflow

Separate experimentation from full retraining.

Two modes:

1. Search mode
   - train from materialized feature tables
   - allow configurable row caps or sampling
   - optimize quickly over many candidate changes
2. Final evaluation mode
   - retrain shortlisted candidates on larger or full training data
   - evaluate on fixed holdout splits
   - generate production artifacts

### D. Experiment Loop

Keep agent-driven loops, but only over task-local model code.

Each loop may modify only:

- feature selection
- derived features from existing columns
- preprocessing
- estimator family
- hyperparameters
- ensemble structure

Each loop may not modify:

- raw-data preprocessing definitions
- split logic
- labels
- warehouse SQL contracts
- evaluation rules

### E. Metrics

Primary metrics:

- short-horizon: `precision_at_top_k`
- chronic: mean of `precision_at_top_k` across 10% and 20%
- lesson: `precision_at_top_k`

Secondary metrics:

- PR-AUC
- Brier score
- subgroup metrics by stage
- calibration diagnostics
- row counts and positive rates

Promotion gates must not rely on a single metric.

### F. Deployment

For production scoring:

- prefer SQL-exportable or otherwise low-latency models for short-horizon and chronic
- evaluate lesson model deployment separately due to scale

Deployment outputs should support:

- Python batch scoring
- SQL scoring artifact where supported
- reproducible feature schema and transform logic

## Recommended Architecture

### 1. Warehouse or Feature Build Layer

Use SQL, dbt, Spark, or warehouse-native transforms to produce model-ready tables.

Responsibilities:

- joins across attendance, schedule, and calendar
- rolling and cumulative features
- task-specific target construction
- train, validation, and test split tagging
- snapshotting and versioning feature tables

Output:

- parquet extracts or direct warehouse tables for training jobs

### 2. Training Layer

Python training jobs consume feature tables.

Responsibilities:

- fit candidate models
- evaluate fixed metrics
- log structured results
- save model artifacts
- optionally export SQL for supported models

### 3. Agent Layer

Codex modifies experiment code and reviews compact outputs.

Responsibilities:

- edit training scripts
- tune model logic
- inspect summarized metrics
- propose next candidates

Not responsible for:

- iterating over raw data interactively
- handling huge logs or full-table dumps
- serving as the execution engine for dozens of full-data runs

## Required System Components

### 1. Feature Materialization Jobs

Deliverables:

- SQL or dbt models, or equivalent jobs, for all task feature tables
- reproducible schema definitions
- data validation checks
- row-count and null-rate summaries

### 2. Training Entry Points

Deliverables:

- `train_short_horizon.py`
- `train_chronic.py`
- `train_lesson.py`

Each must accept:

- feature table or parquet path
- split selection
- row cap or sample fraction
- random seed
- output directory

### 3. Search Loop Entry Points

Deliverables:

- loop scaffolding per task
- search mode config
- results logging
- report generation
- safe experiment isolation

### 4. Final Evaluation Jobs

Deliverables:

- full-data retraining path
- holdout evaluation path
- subgroup evaluation
- calibration report
- production artifact packaging

### 5. SQL Export Path

Required first for:

- short-horizon
- chronic

Optional or deferred for:

- lesson

Export artifacts must include:

- SQL expression or model bundle
- expected input schema
- reference metrics
- complexity notes such as SQL size and runtime

## Performance Requirements

### Search Mode

Target practical iteration times:

- short-horizon: minutes, not hours
- chronic: minutes, not hours
- lesson: acceptable on capped or sampled data; full-data search is not required

### Full Evaluation Mode

Accept slower runs for shortlisted candidates, but they must be batch-job based, not chat-driven.

### Resource Controls

Must support:

- maximum training rows per task
- positive or negative sampling ratio
- optional stage-balanced sampling
- optional downsampling for lesson rows
- cached feature loads

## Experiment Strategy

### Search Phase

Use smaller but representative training subsets.

Rules:

- maintain fixed validation split
- keep same feature schema
- compare many variants cheaply
- log all experiments in structured form

### Promotion Phase

Only top candidates advance.

Rules:

- rerun on larger or full training data
- evaluate across multiple historical splits
- require no regression on key secondary metrics
- require acceptable subgroup behavior

## Governance Rules

A model may be promoted only if:

1. Primary metric improves meaningfully.
2. PR-AUC does not materially regress.
3. Brier or calibration does not materially regress.
4. Subgroup performance remains acceptable.
5. Runtime and scoring complexity remain acceptable.
6. SQL export or production scoring path is validated if required.

## Reporting Requirements

Each final candidate should generate a compact report including:

- model family
- feature set summary
- train, validation, and test metrics
- subgroup metrics
- calibration metrics
- permutation importance or interpretable feature ranking
- experiment history
- deployment notes
- SQL complexity notes if exported

## Data Contracts

Each feature table must include at minimum:

- unique row key
- student and school identifiers
- date fields
- academic year
- split label
- target
- stable feature columns
- subgroup columns such as stage

Feature contracts must be versioned so training code can detect schema drift.

## Logging and Artifacts

Every experiment must write structured outputs:

- `metrics.json`
- `results.tsv`
- model artifact
- config snapshot
- feature schema snapshot
- git commit hash
- training sample size
- runtime summary

Avoid relying only on terminal output.

## Recommended Model Strategy

### Short-Horizon

Start with regularized logistic models and simple tree models.

### Chronic

Start with dual logistic models sharing the same feature basis.

### Lesson

Start with simpler, scalable baselines before trying complex ensembles.

Recommendation:

Use SQL-friendly models first. Only introduce more complex tree or boosting models if they show meaningful gains after accounting for deployment cost.

## Codex Usage Policy

Use Codex for:

- generating and editing code
- setting up training and report scripts
- reading compact metrics artifacts
- reviewing candidate changes
- drafting SQL export helpers

Do not use Codex for:

- repeatedly reading full raw datasets
- running 50-100 full-data experiments interactively
- consuming huge logs every round

When using Codex on real data, provide only:

- schema summaries
- sampled rows
- metric summaries
- artifact paths
- error snippets

## Implementation Phases

### Phase 1: Feature Materialization

- define warehouse feature tables
- validate splits and leakage protections
- produce reproducible extracts

### Phase 2: Scalable Training Jobs

- train from feature tables
- add row caps or sampling
- log structured artifacts

### Phase 3: Controlled Experiment Loops

- run agent loops on sampled or capped data
- keep stable validation
- generate reports

### Phase 4: Promotion Pipeline

- rerun top candidates on full data
- evaluate across backtests
- generate production artifacts

### Phase 5: SQL Deployment

- export supported final models to SQL
- benchmark SQL size and runtime
- compare predictive loss vs deployment simplicity

## Acceptance Criteria

1. Raw preprocessing is not rerun inside every experiment loop iteration.
2. All three model families train from precomputed feature tables.
3. Search loops support capped or sample training.
4. Full-data finalists can be retrained outside the interactive loop.
5. Short-horizon and chronic have a validated production scoring path, preferably SQL-capable.
6. Lesson has a documented production scoring path, even if Python-first.
7. Promotion gates use multiple metrics and subgroup checks.
8. Codex can work effectively from compact artifacts instead of raw data.

## Open Decisions To Resolve When Data Arrives

1. Warehouse platform: dbt, Snowflake, BigQuery, Spark, or other
2. Desired scoring environment: Python batch, SQL, or both
3. Maximum acceptable scoring latency
4. Whether lesson model must be SQL-deployable
5. Acceptable sample sizes for search mode
6. Promotion thresholds for primary and secondary metrics
7. Required subgroup fairness and performance thresholds

## Implementation Brief for Codex

When real data is available, Codex should be asked to:

1. build warehouse-compatible feature contracts for all three model families
2. refactor training to consume materialized feature tables
3. add sampling and capping controls for search mode
4. keep fixed time-based evaluation splits
5. implement structured experiment artifacts and reports
6. add finalist retraining jobs
7. add SQL export for supported final models
8. keep LLM interaction limited to code editing and compact metric review
