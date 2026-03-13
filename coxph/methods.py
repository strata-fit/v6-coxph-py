import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import chi2, norm
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import error, info, warn
from v6_federated_core import (
    MethodContext,
    MethodRegistry,
    MethodSpec,
    PartialFailureError,
)

from .contracts import (
    CentralInput,
    CentralOutput,
    ComputeSummedZInput,
    ComputeSummedZOutput,
    GetUniqueEventTimesInput,
    GetUniqueEventTimesOutput,
    PerformIterationInput,
    PerformIterationOutput,
)

MAX_N_THRESHOLD_RETRIES = 3
LARGE_VALUE_WARNING_THRESHOLD = 10.0


def _get_client(context: MethodContext) -> AlgorithmClient:
    client = context.meta.get("client")
    if client is None:
        raise RuntimeError("Method context is missing the AlgorithmClient")
    return client


def _get_dataframe(context: MethodContext) -> pd.DataFrame:
    df = context.meta.get("df")
    if df is None:
        raise RuntimeError("Method context is missing the dataframe")
    return df


def _parse_partial_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result, dict) and result.get("ok") is False:
        raise PartialFailureError(
            "Partial task returned a failure envelope",
            meta={"error_count": len(result.get("errors", []))},
        )
    if isinstance(result, dict) and result.get("ok") is True:
        payload = result.get("payload")
        return payload if isinstance(payload, dict) else {}
    return result


def _run_partial_task(
    client: AlgorithmClient,
    input_: Dict[str, Any],
    organization_ids: List[int],
    *,
    name: str,
    description: str,
) -> List[Dict[str, Any]]:
    if not organization_ids:
        raise RuntimeError("No organizations available for partial task dispatch")

    info(
        f"Creating partial task '{input_.get('method')}' for "
        f"{len(organization_ids)} organizations."
    )
    task = client.task.create(
        input_=input_,
        organizations=organization_ids,
        name=name,
        description=description,
    )
    info(f"Waiting for partial task results (task_id={task['id']})")
    results = client.wait_for_results(task_id=task["id"])
    return [_parse_partial_result(result) for result in results]


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warn("Matrix inversion failed; using pseudo-inverse")
        return np.linalg.pinv(matrix)


def _to_numpy_beta(values: pd.Series | np.ndarray | List[float]) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _compute_log_likelihood(
    z_sum: pd.Series,
    beta: np.ndarray,
    summed_agg1: np.ndarray,
    aggregated_time_events: pd.DataFrame,
) -> float:
    linear_part = float(np.dot(z_sum.to_numpy(dtype=float), beta))
    risk_set_part = 0.0
    for i, row in aggregated_time_events.iterrows():
        if i >= len(summed_agg1):
            break
        denom = float(summed_agg1[i])
        if denom <= 0.0:
            warn(f"Risk set denominator non-positive at index {i}: {denom}")
            continue
        risk_set_part += float(row["freq"]) * math.log(denom)
    return linear_part - risk_set_part


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series | np.ndarray | List[float],
) -> tuple[np.ndarray, np.ndarray]:
    z_sum_vec = _to_numpy_beta(z_sum)
    n_covariates = z_sum_vec.shape[0]
    tot_p1 = np.zeros(n_covariates, dtype=float)
    tot_p2 = np.zeros((n_covariates, n_covariates), dtype=float)

    for index, row in aggregated_time_events.iterrows():
        denom = float(summed_agg1[index])
        if denom <= 0.0:
            continue
        freq = float(row["freq"])
        s1 = freq * (summed_agg2[index] / denom)
        first_part = summed_agg3[index] / denom
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        second_part = numerator / (denom * denom)
        s2 = freq * (first_part - second_part)
        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum_vec - tot_p1
    secondary_derivative = -tot_p2
    return primary_derivative, secondary_derivative


def _build_results_table(
    beta: np.ndarray,
    s_errors: np.ndarray,
    expl_vars: List[str],
) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.divide(
            beta,
            s_errors,
            out=np.zeros_like(beta, dtype=float),
            where=s_errors > 0,
        )
    pvalues = 2.0 * norm.sf(np.abs(zvalues))

    results = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "Exp(coef)": np.around(np.exp(beta), 5),
            "SE": np.around(s_errors, 5),
            "Var": expl_vars,
            "Z": zvalues,
            "p-value": pvalues,
        }
    )
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    return results.set_index("Var")


def _resolve_organization_ids(
    client: AlgorithmClient,
    organization_ids: Optional[List[int]],
    context: MethodContext,
) -> List[int]:
    if isinstance(organization_ids, list):
        return list(dict.fromkeys(organization_ids))
    if context.organization_ids:
        return list(dict.fromkeys(context.organization_ids))
    organizations = client.organization.list()
    return [organization["id"] for organization in organizations]


def central_handler(
    data: CentralInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for the central handler")

    client = _get_client(context)
    ids = _resolve_organization_ids(client, data.organization_ids, context)
    excluded_ids: List[int] = []

    info(f"Sending task to organizations {ids}")

    n_covs = len(data.expl_vars)
    max_iterations = max(1, int(data.max_iterations))
    tolerance = float(data.tolerance)

    unique_time_events: List[float] = []
    aggregated_time_events = pd.DataFrame(columns=[data.time_col, "freq"])

    n_loops = 0
    while True:
        if n_loops >= MAX_N_THRESHOLD_RETRIES:
            error("Sample size threshold could not be met after retries")
            raise ValueError("Sample size threshold could not be met after retries")
        n_loops += 1
        loop_excluded: List[int] = []

        results = _run_partial_task(
            client,
            input_={
                "method": "get_unique_event_times",
                "kwargs": {
                    "time_col": data.time_col,
                    "outcome_col": data.outcome_col,
                },
            },
            organization_ids=ids,
            name="Unique event times",
            description="Get unique event times and frequencies",
        )

        unique_frames: List[pd.DataFrame] = []
        for output in results:
            not_met = output.get("n_threshold_not_met")
            if not_met is not None:
                warn(
                    f"Insufficient samples for organization {not_met}; "
                    "excluding organization from analysis."
                )
                if not_met in ids:
                    ids.remove(not_met)
                excluded_ids.append(not_met)
                loop_excluded.append(not_met)
                continue

            times = output.get("times")
            if times:
                unique_frames.append(pd.DataFrame.from_dict(times))

        if loop_excluded and not ids:
            warn("No organizations meet the minimal sample size threshold")
            return {
                "included_organizations": [],
                "excluded_organizations": excluded_ids,
                "table": float("nan"),
                "warnings": ["No organizations met the minimum event threshold"],
            }

        if not loop_excluded:
            if unique_frames:
                aggregated_time_events = pd.concat(unique_frames)
                aggregated_time_events = (
                    aggregated_time_events.groupby(data.time_col, as_index=False).sum()
                )
                unique_time_events = aggregated_time_events[data.time_col].tolist()
            break

    z_results = _run_partial_task(
        client,
        input_={
            "method": "compute_summed_z",
            "kwargs": {
                "outcome_col": data.outcome_col,
                "expl_vars": data.expl_vars,
            },
        },
        organization_ids=ids,
        name="Summed Z statistic",
        description="Compute summed z statistic",
    )

    z_sum = pd.Series(0.0, index=data.expl_vars)
    for output in z_results:
        z_sum += pd.Series(output["sum"], index=data.expl_vars, dtype=float).fillna(0.0)

    beta = np.zeros(n_covs, dtype=float)
    secondary_derivative = -np.eye(n_covs, dtype=float)
    summed_agg1 = np.zeros(len(unique_time_events), dtype=float)

    for _ in range(max_iterations):
        results = _run_partial_task(
            client,
            input_={
                "method": "perform_iteration",
                "kwargs": {
                    "time_col": data.time_col,
                    "expl_vars": data.expl_vars,
                    "beta": beta.tolist(),
                    "unique_time_events": unique_time_events,
                },
            },
            organization_ids=ids,
            name="Cox iteration",
            description="Iterating to find the optimal beta",
        )

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times, dtype=float)
        summed_agg2 = np.zeros((n_times, n_covs), dtype=float)
        summed_agg3 = np.zeros((n_times, n_covs, n_covs), dtype=float)

        for output in results:
            summed_agg1 += np.asarray(output["agg1"], dtype=float)

            agg2_df = pd.DataFrame.from_dict(output["agg2"])
            agg2_df = agg2_df.reindex(columns=data.expl_vars)
            summed_agg2 += agg2_df.to_numpy(dtype=float)

            summed_agg3 += np.asarray(output["agg3"], dtype=float)

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1=summed_agg1,
            summed_agg2=summed_agg2,
            summed_agg3=summed_agg3,
            aggregated_time_events=aggregated_time_events,
            z_sum=z_sum,
        )

        beta_old = beta.copy()
        try:
            beta = beta_old - solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError:
            warn("Hessian is singular; falling back to pseudo-inverse update")
            beta = beta_old - _safe_inverse(secondary_derivative).dot(primary_derivative)

        delta = float(np.max(np.abs(beta - beta_old)))
        if math.isnan(delta):
            warn("Optimization update produced NaN delta; stopping iterations")
            break
        if delta <= tolerance:
            info("Betas have settled; optimization converged")
            break

    fisher = _safe_inverse(-secondary_derivative)
    s_errors = np.sqrt(np.clip(np.diag(fisher), a_min=0.0, a_max=None))

    information = -secondary_derivative
    wald_statistic = float(beta @ information @ beta)
    overall_p_value = float(chi2.sf(wald_statistic, len(beta)))

    aic: Optional[float]
    try:
        log_likelihood = _compute_log_likelihood(
            z_sum=z_sum,
            beta=beta,
            summed_agg1=summed_agg1,
            aggregated_time_events=aggregated_time_events,
        )
        if math.isnan(log_likelihood) or math.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        aic = float(-2.0 * log_likelihood + 2.0 * len(beta))
    except (ValueError, FloatingPointError) as exc:
        warn(f"Could not compute AIC due to numerical/data issue: {exc}")
        aic = float("nan")

    results = _build_results_table(beta=beta, s_errors=s_errors, expl_vars=data.expl_vars)

    model_warnings: List[str] = []
    for covariate, row in results.iterrows():
        coef = float(row["Coef"])
        se = float(row["SE"])
        if (
            abs(coef) > LARGE_VALUE_WARNING_THRESHOLD
            or np.isinf(coef)
            or np.isnan(coef)
            or abs(se) > LARGE_VALUE_WARNING_THRESHOLD
            or np.isinf(se)
            or np.isnan(se)
        ):
            msg = (
                f"Warning: Covariate '{covariate}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            model_warnings.append(msg)

    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results.to_json(),
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": int(len(beta)),
        "warnings": model_warnings,
    }


def get_unique_event_times_handler(
    data: GetUniqueEventTimesInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for get_unique_event_times")

    df = _get_dataframe(context)
    client = context.meta.get("client")

    info("Computing unique event times")
    if int(df[data.outcome_col].notnull().sum()) <= int(data.minimum_events):
        org_id = getattr(client, "organization_id", -1)
        warn("Sub-task skipped because the number of samples is too small")
        return {"n_threshold_not_met": int(org_id)}

    times = df[df[data.outcome_col] == 1].groupby(data.time_col, as_index=False).count()
    times = times.sort_values(by=data.time_col)[[data.time_col, data.outcome_col]]
    times["freq"] = times[data.outcome_col]
    times = times.drop(columns=data.outcome_col)
    return {"times": times.to_dict()}


def compute_summed_z_handler(
    data: ComputeSummedZInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for compute_summed_z")

    df = _get_dataframe(context)
    info("Computing summed z statistics")
    z_sum = df[df[data.outcome_col] == 1][data.expl_vars].sum().astype(float).to_dict()
    return {"sum": z_sum}


def perform_iteration_handler(
    data: PerformIterationInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for perform_iteration")

    df = _get_dataframe(context)
    info("Computing aggregates for the derivation of the partial likelihood")

    beta = np.asarray(data.beta, dtype=float)
    working = df[[data.time_col, *data.expl_vars]].dropna(how="any")
    X = working[data.expl_vars].to_numpy(dtype=float)
    times = working[data.time_col].to_numpy(dtype=float)

    if X.shape[0] == 0:
        zeros = np.zeros((len(data.unique_time_events), len(data.expl_vars)))
        return {
            "agg1": [0.0] * len(data.unique_time_events),
            "agg2": pd.DataFrame(zeros, columns=data.expl_vars).to_dict(),
            "agg3": [
                np.zeros((len(data.expl_vars), len(data.expl_vars))).tolist()
                for _ in data.unique_time_events
            ],
        }

    exp_xb = np.exp(X @ beta)
    agg1: List[float] = []
    agg2_rows: List[np.ndarray] = []
    agg3: List[np.ndarray] = []

    n_covariates = len(data.expl_vars)

    for unique_time in data.unique_time_events:
        mask = times >= float(unique_time)
        if not np.any(mask):
            agg1.append(0.0)
            agg2_rows.append(np.zeros(n_covariates, dtype=float))
            agg3.append(np.zeros((n_covariates, n_covariates), dtype=float))
            continue

        Xi = X[mask]
        exp_i = exp_xb[mask]
        weighted = Xi * exp_i[:, None]

        agg1.append(float(exp_i.sum()))
        agg2_rows.append(weighted.sum(axis=0))
        agg3.append(Xi.T @ weighted)

    agg2_df = pd.DataFrame(agg2_rows, columns=data.expl_vars)
    return {
        "agg1": agg1,
        "agg2": agg2_df.to_dict(),
        "agg3": [matrix.tolist() for matrix in agg3],
    }


METHOD_REGISTRY = MethodRegistry(
    [
        MethodSpec(
            name="central",
            input_model=CentralInput,
            output_model=CentralOutput,
            handler=central_handler,
        ),
        MethodSpec(
            name="get_unique_event_times",
            input_model=GetUniqueEventTimesInput,
            output_model=GetUniqueEventTimesOutput,
            handler=get_unique_event_times_handler,
        ),
        MethodSpec(
            name="compute_summed_z",
            input_model=ComputeSummedZInput,
            output_model=ComputeSummedZOutput,
            handler=compute_summed_z_handler,
        ),
        MethodSpec(
            name="perform_iteration",
            input_model=PerformIterationInput,
            output_model=PerformIterationOutput,
            handler=perform_iteration_handler,
        ),
    ]
)
