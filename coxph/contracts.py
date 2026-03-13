from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CentralInput(BaseModel):
    time_col: str
    outcome_col: str
    expl_vars: List[str] = Field(min_length=1)
    organization_ids: Optional[List[int]] = None
    database_labels: Optional[List[str]] = None
    max_iterations: int = 10
    tolerance: float = 1e-6


class CentralOutput(BaseModel):
    included_organizations: List[int] = Field(default_factory=list)
    excluded_organizations: List[int] = Field(default_factory=list)
    model: Optional[str] = None
    overall_p_value: Optional[float] = None
    aic: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    table: Optional[float] = None


class GetUniqueEventTimesInput(BaseModel):
    time_col: str
    outcome_col: str
    minimum_events: int = 10


class GetUniqueEventTimesOutput(BaseModel):
    times: Optional[Dict[str, Dict[Any, Any]]] = None
    n_threshold_not_met: Optional[int] = None


class ComputeSummedZInput(BaseModel):
    outcome_col: str
    expl_vars: List[str] = Field(min_length=1)


class ComputeSummedZOutput(BaseModel):
    sum: Dict[str, float]


class PerformIterationInput(BaseModel):
    time_col: str
    expl_vars: List[str] = Field(min_length=1)
    beta: List[float]
    unique_time_events: List[float]


class PerformIterationOutput(BaseModel):
    agg1: List[float]
    agg2: Dict[str, Dict[Any, float]]
    agg3: List[List[List[float]]]
