# Infrastructure Testing Manifest (Portable)

This file is the single source of truth for infra setup, smoke execution, and
latest validated results.

## Pinned versions used in latest successful run

- Algorithm repo branch: `feat/abstractions_poc`
- Tested algorithm commit (full hash): `17c410d6d9241b10a4a71a750eaf87d4318e2258`
- Upstream base commit (full hash): `340a85504face6fef5c162a011f7fcee032fddaf`
- Infra harness commit (full hash): `3133deb74a30fe34617d69d94628bbff38c71869`
- Vantage6 version: `4.13.3`

## 1) Mock + math validation

```bash
source /home/debian/Desktop/code/.venv/bin/activate
cd /home/debian/Desktop/code/coxph
pip install -e .
python -m pytest -q test/test.py test/test_math_correctness.py
```

## 2) Local infrastructure smoke (single-file workflow)

### 2.1 Clone and pin infra harness

```bash
cd /home/debian/Desktop/code/coxph
ALG_ROOT="$(pwd)"
INFRA_DIR="$ALG_ROOT/tools/v6-infra"

git clone https://github.com/mdw-nl/v6-infrastructure-sh.git "$INFRA_DIR"
cd "$INFRA_DIR"
git checkout 3133deb74a30fe34617d69d94628bbff38c71869
```

### 2.2 Create node + config files inline

```bash
cd "$ALG_ROOT"

cat > /tmp/coxph_nodes.env <<EOF_NODES
alpha|844a7d92-1cc9-4856-bf33-0613252d5b3c|$ALG_ROOT/test/HEAD-NECK-RADIOMICS-HN1.csv|csv|default
beta|57143784-19ef-456b-94c9-ba68c8cb079b|$ALG_ROOT/test/HEAD-NECK-RADIOMICS-HN1.csv|csv|default
gamma|57143784-19ef-456b-94c9-ba68c8cb079c|$ALG_ROOT/test/HEAD-NECK-RADIOMICS-HN1.csv|csv|default
EOF_NODES

cat > "$INFRA_DIR/infrastructure/config.env" <<EOF_CFG
ENVIRONMENT=CI
PYTHON_INTERPRETER=python3.12
VERSION_VANTAGE6=4.13.3
UI_ENABLED=false
SERVER_URL=http://host.docker.internal
DOCKER_REGISTRY=harbor2.vantage6.ai/infrastructure
STRICT_DATA_CHECKS=true
COLLABORATION_NAME=coxph-ci
NODES_CONFIG=/tmp/coxph_nodes.env
EOF_CFG
```

If `python3.12` is unavailable locally, append:

```bash
echo "PYTHON_INTERPRETER=/home/debian/Desktop/code/.venv/bin/python" >> "$INFRA_DIR/infrastructure/config.env"
echo "VENV_PATH=/home/debian/Desktop/code/.venv" >> "$INFRA_DIR/infrastructure/config.env"
```

### 2.3 Start infra, publish local image, and run task

```bash
cd "$INFRA_DIR/infrastructure"
./infra.sh preflight
ENVIRONMENT=CI UI_ENABLED=false ./infra.sh up

cd "$ALG_ROOT"
docker run -d --restart unless-stopped -p 5001:5000 --name v6-local-registry registry:2 || true
docker build -t localhost:5001/v6-coxph:local .
docker push localhost:5001/v6-coxph:local
```

Submit and validate a smoke task:

```bash
source /home/debian/Desktop/code/.venv/bin/activate
python - <<'PY'
import base64
import json
import time
from io import StringIO

import pandas as pd
from vantage6.client import Client

terminal = {"completed", "crashed", "failed", "cancelled", "non-existing Docker image"}
client = Client("http://localhost", 5070, "/api")
client.authenticate("gamma-user", "gamma-password")
client.setup_encryption(None)

collab = next(c for c in client.collaboration.list()["data"] if c["name"] == "coxph-ci")
org_map = {o["name"]: o["id"] for o in client.organization.list()["data"]}
org_ids = [org_map[n] for n in ["alpha", "beta", "gamma"]]

input_ = {
    "master": True,
    "method": "central",
    "kwargs": {
        "time_col": "overall_survival_in_days",
        "outcome_col": "event_overall_survival",
        "expl_vars": ["clin_n_1", "index_tumour_location_oropharynx"],
        "organization_ids": org_ids,
        "max_iterations": 20,
        "tolerance": 1e-6,
    },
}

task = client.task.create(
    collaboration=collab["id"],
    organizations=[org_map["gamma"]],
    name="coxph-inline-smoke",
    image="localhost:5001/v6-coxph:local",
    description="coxph smoke",
    input_=input_,
    databases=[{"label": "default"}],
)

tid = task["id"]
status = None
for _ in range(600):
    status = client.task.get(tid).get("status")
    if status in terminal:
        break
    time.sleep(2)

if status != "completed":
    raise RuntimeError(f"Task {tid} finished with status {status}")

rows = client.result.from_task(task_id=tid).get("data", [])
if not rows:
    raise RuntimeError(f"Task {tid} has no results")

result = rows[0].get("result")
try:
    decoded = json.loads(base64.b64decode(result).decode("utf-8"))
except Exception:
    decoded = json.loads(result) if isinstance(result, str) else result

if isinstance(decoded, dict) and decoded.get("ok") is False:
    raise RuntimeError(f"Failure envelope: {decoded}")

required = {"aic", "degrees_of_freedom", "excluded_organizations", "included_organizations", "model", "overall_p_value", "warnings"}
missing = required - set(decoded.keys())
if missing:
    raise RuntimeError(f"Missing keys: {sorted(missing)}")

model_df = pd.read_json(StringIO(decoded["model"]))
print("task_id:", tid)
print("status:", status)
print("coef:", model_df["Coef"].to_dict())
print("validation: PASS")
PY
```

### 2.4 Infra smoke + teardown

```bash
cd "$INFRA_DIR/infrastructure"
UI_ENABLED=false ./infra.sh test
ENVIRONMENT=CI UI_ENABLED=false ./infra.sh down

docker rm -f v6-local-registry || true
rm -f /tmp/coxph_nodes.env
```

## 3) Latest run result snapshot

- Date: `2026-03-13T04:49:54-07:00`
- Mock/math tests: `3 passed, 2 warnings`
- Infra preflight/up: passed
- Algorithm smoke task: passed (`completed`)
- Infra `./infra.sh test`: passed
- Teardown: passed

Math correctness snapshot:

- reference beta: `[0.3235030266, 0.0637281462]`
- federated beta: `[0.3235, 0.06373]`
- absolute delta: `[3.03e-06, 1.85e-06]`
