# Retail Analytics Copilot

## Graph Design

- **Router** → Classifies query type (rag/sql/hybrid) using DSPy classifier
- **Retriever** → Fetches top-3 doc chunks with TF-IDF similarity scores and chunk IDs
- **Planner** → Extracts date ranges, KPI formulas, and categories from docs
- **NL→SQL** → Generates SQLite queries using DSPy with live schema (PRAGMA)
- **Executor** → Runs SQL, captures columns/rows/errors
- **Repair** → Fixes failed SQL (max 2 iterations)
- **Synthesizer** → Produces typed answers matching format_hint with citations

Repair loop: SQL error → Repair → Executor (up to 2x) → Synthesizer

## DSPy Module Optimized

**NL→SQL** (Natural Language to SQL generation)

Optimized with **BootstrapFewShot** on 8 training examples

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Avg Score | 0.42 | 0.75 | **+0.33 (+79%)** |
| Valid SQL | 50% | 88% | +38% |
| Execution Success | 38% | 75% | +37% |

## Trade-offs & Assumptions **First Point is VERY IMPORTANT**

- **Date Range**: Database has 2012-2023 data (not 1997), so I updated the sample questions to use 2013 instead, if still used 1997 all answers will be wrong and based on assumptions, also updated marketing_calendar.md to year 2013.
- **CostOfGoods**: Approximated as `0.7 * UnitPrice` (no cost field in Northwind DB)
- **TF-IDF**: Used over embeddings (no dependencies, fast for 8 chunks)
- **Repair Bound**: Max 2 iterations to prevent loops
