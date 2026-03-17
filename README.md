# v6-coxph

Federated Cox proportional hazards algorithm for vantage6 (tested on `4.13.3`).

This repository now follows the same pattern as the linear/imputation repos:

- typed method contracts (`coxph/contracts.py`)
- method registry + handlers (`coxph/methods.py`)
- thin V6 wrappers (`coxph/central.py`, `coxph/partial.py`)
- reusable core math functions (`compute_derivatives` and Newton-Raphson flow)

## Supported methods

- `central`
- `get_unique_event_times`
- `compute_summed_z`
- `perform_iteration`

## Quick start (mock)

```bash
source .venv/bin/activate
pip install -e .
python test/test.py
```

## Infra smoke tests

Infra instructions and latest executed result manifest are documented in:

- `tests/infrastructure_testing.md`

## Math validation

Mathematical validation checks are implemented in:

- `test/test_math_correctness.py`

It compares the federated Newton-Raphson estimates with a direct centralized
optimizer of the Cox partial log-likelihood.
