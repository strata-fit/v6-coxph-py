from pathlib import Path
from io import StringIO

import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

CURRENT_PATH = Path(__file__).parent


def build_client() -> MockAlgorithmClient:
    dataset = {
        "database": CURRENT_PATH / "HEAD-NECK-RADIOMICS-HN1.csv",
        "db_type": "csv",
        "input_data": {},
    }
    return MockAlgorithmClient(
        datasets=[[dataset], [dataset]],
        module="coxph",
    )


def test_central_smoke() -> None:
    client = build_client()
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "overall_survival_in_days",
                "outcome_col": "event_overall_survival",
                "expl_vars": ["clin_n_1", "index_tumour_location_oropharynx"],
                "organization_ids": org_ids,
            },
        },
        organizations=[org_ids[0]],
    )
    results = client.wait_for_results(task.get("id"))
    assert len(results) == 1

    payload = results[0]
    assert "model" in payload
    assert "overall_p_value" in payload
    assert "aic" in payload
    assert payload["included_organizations"] == org_ids
    assert payload["excluded_organizations"] == []

    model_df = pd.read_json(StringIO(payload["model"]))
    assert list(model_df.index) == ["clin_n_1", "index_tumour_location_oropharynx"]
    assert {"Coef", "Exp(coef)", "SE", "Z", "p-value"}.issubset(model_df.columns)


def test_central_smoke_legacy_argument_names() -> None:
    client = build_client()
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_column_name": "overall_survival_in_days",
                "outcome_column_name": "event_overall_survival",
                "predictors": ["clin_n_1", "index_tumour_location_oropharynx"],
                "organization_ids": org_ids,
            },
        },
        organizations=[org_ids[0]],
    )
    results = client.wait_for_results(task.get("id"))
    assert len(results) == 1
    assert "model" in results[0]


if __name__ == "__main__":
    test_central_smoke()
    print("mock smoke test passed")
