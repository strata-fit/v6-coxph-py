from typing import Any, Dict, List

import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY


@data(1)
@algorithm_client
def get_unique_event_times(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    minimum_events: int = 10,
) -> Dict[str, Any]:
    """Thin V6 adapter for the typed get_unique_event_times method."""
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "get_unique_event_times",
        {
            "time_col": time_col,
            "outcome_col": outcome_col,
            "minimum_events": minimum_events,
        },
        context=MethodContext(
            method="get_unique_event_times",
            meta={"df": df, "client": client},
        ),
    )
    return to_v6_result(envelope)


@data(1)
def compute_summed_z(
    df: pd.DataFrame,
    outcome_col: str,
    expl_vars: List[str],
) -> Dict[str, Any]:
    """Thin V6 adapter for the typed compute_summed_z method."""
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "compute_summed_z",
        {
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        },
        context=MethodContext(
            method="compute_summed_z",
            meta={"df": df},
        ),
    )
    return to_v6_result(envelope)


@data(1)
def perform_iteration(
    df: pd.DataFrame,
    time_col: str,
    expl_vars: List[str],
    beta: List[float],
    unique_time_events: List[float],
) -> Dict[str, Any]:
    """Thin V6 adapter for the typed perform_iteration method."""
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "perform_iteration",
        {
            "time_col": time_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "unique_time_events": unique_time_events,
        },
        context=MethodContext(
            method="perform_iteration",
            meta={"df": df},
        ),
    )
    return to_v6_result(envelope)
