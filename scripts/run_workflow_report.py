from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:5000"
ADMIN = {"username": "admin@ivx.local", "password": "Admin@12345"}
TEST_USER = {
    "username": "training_user_20260808",
    "email": "training_user_20260808@example.test",
    "password": "TrainUser@12345",
}
MODEL_ID = "test-hybrid-7b"
DATASET_ID = "94efed55-6d c6-5619-80c3-72e73d8be976".replace(" ", "")
REPORT_PATH = Path("docs/training-workflow-report.md")


def csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    if not match:
        raise RuntimeError("CSRF token not found")
    return match.group(1)


def login(session: requests.Session, username: str, password: str) -> requests.Response:
    page = session.get(f"{BASE_URL}/login")
    response = session.post(
        f"{BASE_URL}/login",
        data={"username": username, "password": password, "csrf_token": csrf(page.text)},
        allow_redirects=True,
    )
    if response.status_code != 200 or "/login" in response.url:
        raise RuntimeError(f"Login failed for {username}: {response.status_code} {response.url}")
    return response


def main() -> None:
    started = datetime.now(timezone.utc)
    admin_session = requests.Session()
    login(admin_session, **ADMIN)
    admin_page = admin_session.get(f"{BASE_URL}/admin/users")
    create = admin_session.post(
        f"{BASE_URL}/admin/users",
        data={
            **TEST_USER,
            "role": "member",
            "csrf_token": csrf(admin_page.text),
        },
        allow_redirects=True,
    )
    user_created = "provisioned successfully" in create.text
    admin_session.post(
        f"{BASE_URL}/logout",
        data={"csrf_token": csrf(create.text)},
        allow_redirects=True,
    )

    user_session = requests.Session()
    login(user_session, TEST_USER["username"], TEST_USER["password"])
    training_page = user_session.get(f"{BASE_URL}/training/new")
    training = user_session.post(
        f"{BASE_URL}/training/new",
        data={
            "name": "AFE Hybrid 7B Chat Validation Run",
            "model_id": MODEL_ID,
            "dataset_id": DATASET_ID,
            "max_steps": "20",
            "learning_rate": "0.0001",
            "batch_size": "2",
            "precision": "fp16",
            "csrf_token": csrf(training_page.text),
        },
        allow_redirects=True,
    )
    match = re.search(r"/training/([0-9a-f-]+)", training.url)
    if not match:
        raise RuntimeError(f"Training was not queued: {training.status_code} {training.url}")
    experiment_id = match.group(1)

    live = {}
    for _ in range(60):
        live = user_session.get(f"{BASE_URL}/training/{experiment_id}/api/live").json()
        if live.get("status") in {"completed", "failed"}:
            break
        time.sleep(0.25)
    if live.get("status") != "completed":
        raise RuntimeError(f"Training did not complete: {live}")

    evaluation_page = user_session.get(f"{BASE_URL}/evaluation/new")
    evaluation = user_session.post(
        f"{BASE_URL}/evaluation/new",
        data={
            "benchmark_name": "mmlu",
            "model_id": MODEL_ID,
            "csrf_token": csrf(evaluation_page.text),
        },
        allow_redirects=True,
    )
    evaluation_queued = "queued in background" in evaluation.text

    report = f"""# Local 7B Chat Training Workflow Report

**Run date:** {started.astimezone().isoformat(timespec='seconds')}  
**Environment:** Local Flask application at `{BASE_URL}`  
**Workflow status:** Completed through training and evaluation queue submission

## Steps Executed

1. Seeded or verified the default superadmin with `python scripts/seed_admin.py`.
2. Logged in as `admin@ivx.local`.
3. Provisioned `{TEST_USER['username']}` with the `member` role.
4. Logged out and logged in as the provisioned user.
5. Selected the existing compiled model `AFE Hybrid 7B Base` (`{MODEL_ID}`), reported by the database as approximately 9.30B parameters.
6. Selected the ready `TinyStories sample` dataset (`{DATASET_ID}`).
7. Submitted `AFE Hybrid 7B Chat Validation Run` with 20 steps, batch size 2, learning rate 0.0001, and FP16.
8. Polled the live training endpoint until completion.
9. Submitted an MMLU evaluation job for the same model.

## Observed Training Result

| Metric | Value |
|---|---:|
| Experiment ID | `{experiment_id}` |
| Status | `{live.get('status')}` |
| Final step | `{live.get('current_step')}` / `{live.get('max_steps')}` |
| Final loss | `{live.get('current_loss')}` |
| Best loss | `{live.get('best_loss')}` |
| Total tokens | `{live.get('total_tokens')}` |
| Tokens per second | `{live.get('tokens_per_sec')}` |
| Evaluation submission | {'Queued' if evaluation_queued else 'Not confirmed'} |

## Response Validation

The chat response shown in the supplied screenshot is **not a real model response**. It says:

> AI Foundry Enterprise Hybrid Architecture: Processed 'Hey...'. Configure a vLLM or llama.cpp model artifact for real generation.

This is an application fallback/placeholder response. It confirms that the chat route handled the request, but it does **not** validate generation quality, chat correctness, or the trained checkpoint. A genuine response validation requires a configured vLLM or llama.cpp artifact and an inference request routed to that deployed artifact.

## Screen Recording

No screen recording was generated by this automated HTTP run. The workflow was executed through authenticated local HTTP sessions. For an auditable recording, start Windows Game Bar with `Win+Alt+R` or OBS before repeating the browser clicks; do not describe the fallback chat message as model evidence.

## Limitations and Findings

- The repository's `TRAINING_BACKEND=auto` path uses the simulated training loop unless Transformers is explicitly selected and a compatible model artifact is configured.
- The 7B record is an architecture definition with generated config/code; this run did not download or train a real 7B checkpoint.
- MMLU was queued, but this report does not treat a benchmark score as complete until the evaluation job's persisted status and score are checked.
- The TinyStories metadata has an unrelated `license_url` (`https://www.aicpa.org/cdma`) that should be corrected separately.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"experiment_id": experiment_id, "training": live, "evaluation_queued": evaluation_queued, "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
