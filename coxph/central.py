from typing import Any, Dict, List, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    organization_ids: Optional[List[int]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Thin V6 adapter for the typed central CoxPH method."""
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
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "organization_ids": organization_ids,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        },
        context=context,
    )
    return to_v6_result(envelope)
