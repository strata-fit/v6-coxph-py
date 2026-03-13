from typing import Any, Dict, List, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from v6_federated_core import (
    ConfigError,
    MethodContext,
    dispatch_registered_method,
    to_v6_result,
)

from .methods import METHOD_REGISTRY


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    expl_vars: Optional[List[str]] = None,
    predictors: Optional[List[str]] = None,
    time_column_name: Optional[str] = None,
    outcome_column_name: Optional[str] = None,
    organization_ids: Optional[List[int]] = None,
    database_labels: Optional[List[str]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Thin V6 adapter for the typed central CoxPH method."""

    resolved_time_col = time_col or time_column_name
    if time_col and time_column_name and time_col != time_column_name:
        raise ConfigError("Conflicting values for time_col and time_column_name")
    if not resolved_time_col:
        raise ConfigError("Missing required argument: time_col (or time_column_name)")

    resolved_outcome_col = outcome_col or outcome_column_name
    if outcome_col and outcome_column_name and outcome_col != outcome_column_name:
        raise ConfigError("Conflicting values for outcome_col and outcome_column_name")
    if not resolved_outcome_col:
        raise ConfigError(
            "Missing required argument: outcome_col (or outcome_column_name)"
        )

    resolved_expl_vars = expl_vars or predictors
    if expl_vars and predictors and expl_vars != predictors:
        raise ConfigError("Conflicting values for expl_vars and predictors")
    if not resolved_expl_vars:
        raise ConfigError("Missing required argument: expl_vars (or predictors)")

    resolved_ids = organization_ids or [org["id"] for org in client.organization.list()]
    context = MethodContext(
        method="central",
        organization_ids=resolved_ids,
        meta={"client": client},
    )
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "central",
        {
            "time_col": resolved_time_col,
            "outcome_col": resolved_outcome_col,
            "expl_vars": resolved_expl_vars,
            "organization_ids": organization_ids,
            "database_labels": database_labels,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        },
        context=context,
    )
    return to_v6_result(envelope)
