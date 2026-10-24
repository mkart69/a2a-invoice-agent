
import os
import json
import uuid
import hashlib
import threading
from datetime import datetime, timezone

import psycopg
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row

app = FastAPI(title="A2A Invoice Action Agent")

DATABASE_URL = os.getenv("DATABASE_URL", "")
BEARER_TOKEN = os.getenv("A2A_BEARER_TOKEN", "")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv(
    "AI_BASE_URL", "https://api.groq.com/openai/v1"
)
AI_MODEL = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MEDIA = "application/a2a+json"

db_lock = threading.RLock()


def connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg.connect(url, row_factory=dict_row)


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                state TEXT NOT NULL,
                data JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS tasks_principal_idx
            ON tasks(principal)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                package_hash TEXT PRIMARY KEY,
                proposal JSONB NOT NULL
            )
        """)


@app.on_event("startup")
def startup():
    init_db()


def now():
    return datetime.now(timezone.utc).isoformat()


def error(status, code, message):
    return JSONResponse(
        status_code=status,
        media_type=A2A_MEDIA,
        content={"error": {"code": code, "message": message}},
    )


def get_principal(request):
    if not BEARER_TOKEN:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    supplied = auth[7:]
    if not supplied or not hashlib.sha256(
        supplied.encode()
    ).digest() == hashlib.sha256(BEARER_TOKEN.encode()).digest():
        return None
    return hashlib.sha256(supplied.encode()).hexdigest()


def check_headers(request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "INVALID_VERSION", "A2A-Version must be 1.0")
    content_type = request.headers.get("content-type", "")
    if "application/a2a+json" not in content_type:
        return error(415, "UNSUPPORTED_MEDIA_TYPE",
                     "Use application/a2a+json")
    return None


def task_state(task):
    return task["status"]["state"]


def message_id(message):
    return message.get("messageId", "")


def package_digest(package):
    raw = json.dumps(
        package, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_batch(message):
    for part in message.get("parts", []):
        if part.get("mediaType") == INPUT_MODE:
            return part.get("data")
    return None


def get_results(message):
    for part in message.get("parts", []):
        if part.get("mediaType") == RESULT_MODE:
            return part.get("data")
    return None


def make_message(role, media_type, data):
    return {
        "messageId": str(uuid.uuid4()),
        "role": role,
        "parts": [{"mediaType": media_type, "data": data}],
    }


def get_proposal(package, batch_id, policy_revision):
    digest = package_digest(package)

    with connect() as conn:
        row = conn.execute(
            "SELECT proposal FROM decisions WHERE package_hash = %s",
            (digest,),
        ).fetchone()
        if row:
            proposal = row["proposal"]
            proposal = dict(proposal)
            proposal["actionId"] = str(uuid.uuid4()) + str(uuid.uuid4())
            return proposal

    if not AI_API_KEY:
        raise RuntimeError("AI_API_KEY is not configured")

    prompt = """
You are a careful invoice action proposal agent. Analyze the supplied
synthetic invoice package and decide using the actual evidence, not merely
keyword matching. Return ONLY a JSON object with these keys:
vendorName, invoiceNumber, amountMinor, currency, action, evidenceRefs,
rationale.

action must be exactly one of:
settle_invoice, request_approval, hold_invoice, reject_duplicate,
open_exception.

Choose settle_invoice only when the invoice is valid, reconciled, and
within autonomous authority. Choose request_approval when commercially
valid but outside delegated authority. Choose hold_invoice when payment
must pause until a specific verification completes. Choose reject_duplicate
only when the same commercial invoice was already paid. Choose open_exception
when material records conflict.

Use exactly three decisive bracketed evidence references copied verbatim
from the decisive evidence paragraph. Do not use cover pages, archives,
training examples, or decoys as decisive evidence. State the reasoning,
the selected action, and at least two of the selected evidence references.
Rationale must be 60-1500 characters. amountMinor is an integer in minor
currency units. currency must be INR. Never invent missing facts. If the
records conflict, choose open_exception and explain the conflict.
"""

    payload = {
        "model": AI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                {"batchId": batch_id, "policyRevision": policy_revision,
                 "package": package}, ensure_ascii=False
            )},
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        AI_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + AI_API_KEY},
        json=payload,
        timeout=35,
    )
    response.raise_for_status()
    result = json.loads(response.json()["choices"][0]["message"]["content"])

    proposal = {
        "packageId": package.get("packageId", ""),
        "actionId": str(uuid.uuid4()) + str(uuid.uuid4()),
        "action": result["action"],
        "facts": {
            "vendorName": result.get("vendorName", ""),
            "invoiceNumber": result.get("invoiceNumber", ""),
            "amountMinor": result.get("amountMinor", 0),
            "currency": "INR",
        },
        "evidenceRefs": result.get("evidenceRefs", []),
        "rationale": result.get("rationale", ""),
    }

    allowed = {
        "settle_invoice", "request_approval", "hold_invoice",
        "reject_duplicate", "open_exception",
    }
    if proposal["action"] not in allowed:
        raise ValueError("AI returned an unsupported action")
    if not isinstance(proposal["facts"]["amountMinor"], int):
        raise ValueError("AI returned an invalid amount")
    if len(proposal["evidenceRefs"]) != 3:
        raise ValueError("AI must return exactly three evidence references")
    if not 60 <= len(proposal["rationale"]) <= 1500:
        raise ValueError("AI rationale has an invalid length")

    with connect() as conn:
        conn.execute(
            """INSERT INTO decisions(package_hash, proposal)
               VALUES (%s, %s) ON CONFLICT(package_hash) DO NOTHING""",
            (digest, json.dumps(proposal)),
        )
    return proposal


def save_task(task):
    with connect() as conn:
        conn.execute(
            """INSERT INTO tasks(id, principal, state, data)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT(id) DO UPDATE SET
               state = EXCLUDED.state, data = EXCLUDED.data,
               updated_at = NOW()""",
            (task["id"], task["_principal"], task_state(task),
             json.dumps(task)),
        )


def read_task(task_id, principal):
    with connect() as conn:
        row = conn.execute(
            "SELECT data FROM tasks WHERE id=%s AND principal=%s",
            (task_id, principal),
        ).fetchone()
    return row["data"] if row else None


def public_task(task):
    result = dict(task)
    result.pop("_principal", None)
    return result


def initial_task(message, batch, principal):
    batch_id = batch.get("batchId", "")
    packages = batch.get("packages", [])
    if not batch_id or not isinstance(packages, list) or not packages:
        raise ValueError("Invalid invoice claim batch")
    if len(packages) > 100:
        raise ValueError("Batch is too large")

    proposals = []
    seen = set()
    for package in packages:
        pid = package.get("packageId")
        if not pid or pid in seen:
            raise ValueError("Package IDs must be present and unique")
        seen.add(pid)
        proposals.append(get_proposal(
            package, batch_id, batch.get("policyRevision", "")
        ))

    task_id = str(uuid.uuid4())
    proposal_message = make_message(
        "ROLE_AGENT", PROPOSAL_MODE,
        {"batchId": batch_id, "proposals": proposals},
    )
    task = {
        "id": task_id,
        "contextId": str(uuid.uuid4()),
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
            "timestamp": now(),
        },
        "history": [message, proposal_message],
        "artifacts": [{
            "artifactId": str(uuid.uuid4()),
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROPOSAL_MODE,
                       "data": {"batchId": batch_id,
                                "proposals": proposals}}],
        }],
        "_principal": principal,
        "_batchId": batch_id,
        "_proposals": proposals,
    }
    return task


def apply_results(task, message, results_data):
    if task_state(task) != "TASK_STATE_INPUT_REQUIRED":
        raise ValueError("Task is not waiting for results")
    batch_id = task.get("_batchId")
    if results_data.get("batchId") != batch_id:
        raise ValueError("Batch ID does not match")

    proposals = {p["packageId"]: p for p in task["_proposals"]}
    results = results_data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Results must be a list")

    executions = []
    seen = set()
    for result in results:
        pid = result.get("packageId")
        if pid in seen or pid not in proposals:
            raise ValueError("Unknown or repeated package ID")
        seen.add(pid)
        proposal = proposals[pid]
        if (result.get("actionId") != proposal["actionId"]
                or result.get("action") != proposal["action"]):
            raise ValueError("Result does not match the stored proposal")
        if result.get("outcome") not in ("ACCEPTED", "REJECTED"):
            raise ValueError("Invalid result outcome")
        if not result.get("receiptNonce"):
            raise ValueError("Missing receipt nonce")

        if result["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": pid,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": result["receiptNonce"],
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
            })

    if len(results) != len(proposals):
        raise ValueError("Every proposal must have exactly one result")

    task["history"].append(message)
    task["artifacts"].append({
        "artifactId": str(uuid.uuid4()),
        "name": "invoice-action-receipts",
        "parts": [{"mediaType": RECEIPT_MODE,
                   "data": {"batchId": batch_id,
                            "executions": executions}}],
    })
    task["status"] = {
        "state": "TASK_STATE_COMPLETED",
        "timestamp": now(),
    }
    return task


@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "name": "A2A Invoice Action Agent",
        "description": "Proposes safe invoice actions and records accepted results.",
        "version": "1.0.0",
        "supportedInterfaces": [{
            "url": base,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [PROPOSAL_MODE, RECEIPT_MODE],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reviews invoice evidence and proposes controlled actions.",
            "tags": ["invoice", "finance", "approval", "audit"],
            "inputModes": [INPUT_MODE],
            "outputModes": [PROPOSAL_MODE, RECEIPT_MODE],
        }],
    }


@app.post("/message:send")
async def send_message(request: Request):
    auth_problem = check_headers(request)
    if auth_problem:
        return auth_problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")

    try:
        body = await request.json()
        message = body.get("message", {})
        mid = message_id(message)
        if not mid:
            return error(400, "INVALID_MESSAGE", "messageId is required")

        canonical = json.dumps(
            message, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()

        with db_lock:
            with connect() as conn:
                existing = conn.execute(
                    """SELECT body_hash, task_id FROM idempotency
                       WHERE principal=%s AND message_id=%s""",
                    (principal, mid),
                ).fetchone()
            if existing:
                if existing["body_hash"] != digest:
                    return error(409, "IDEMPOTENCY_CONFLICT",
                                 "messageId was used with different content")
                task = read_task(existing["task_id"], principal)
                if task:
                    return JSONResponse(
                        {"task": public_task(task)}, media_type=A2A_MEDIA
                    )

            results_data = get_results(message)
            if results_data:
                task_id = body.get("taskId")
                task = read_task(task_id, principal) if task_id else None
                if not task:
                    return error(404, "TASK_NOT_FOUND", "Task not found")
                try:
                    task = apply_results(task, message, results_data)
                except ValueError as exc:
                    return error(400, "INVALID_RESULTS", str(exc))
            else:
                batch = get_batch(message)
                if not batch:
                    return error(400, "INVALID_MESSAGE",
                                 "Invoice claim batch part is required")
                try:
                    task = initial_task(message, batch, principal)
                except (ValueError, KeyError, requests.RequestException,
                        RuntimeError) as exc:
                    return error(502, "PROCESSING_FAILED",
                                 "Could not safely process this batch")

            task["_principal"] = principal
            save_task(task)
            with connect() as conn:
                conn.execute(
                    """INSERT INTO idempotency
                       (principal, message_id, body_hash, task_id)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT(principal, message_id) DO NOTHING""",
                    (principal, mid, digest, task["id"]),
                )
            return JSONResponse(
                {"task": public_task(task)}, media_type=A2A_MEDIA
            )
    except Exception:
        return error(500, "INTERNAL_ERROR", "Request could not be processed")


@app.get("/tasks")
def list_tasks(request: Request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "INVALID_VERSION", "A2A-Version must be 1.0")
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    with connect() as conn:
        rows = conn.execute(
            "SELECT data FROM tasks WHERE principal=%s ORDER BY created_at DESC",
            (principal,),
        ).fetchall()
    return JSONResponse(
        {"tasks": [public_task(row["data"]) for row in rows]},
        media_type=A2A_MEDIA,
    )


@app.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "INVALID_VERSION", "A2A-Version must be 1.0")
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    task = read_task(task_id, principal)
    if not task:
        return error(404, "TASK_NOT_FOUND", "Task not found")
    return JSONResponse(public_task(task), media_type=A2A_MEDIA)


@app.post("/tasks/{task_id}:cancel")
def cancel_task(task_id: str, request: Request):
    auth_problem = check_headers(request)
    if auth_problem:
        return auth_problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")

    with db_lock:
        task = read_task(task_id, principal)
        if not task:
            return error(404, "TASK_NOT_FOUND", "Task not found")
        if task_state(task) in (
            "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED", "TASK_STATE_FAILED",
        ):
            return error(409, "TASK_TERMINAL", "Task is already terminal")
        task["status"] = {
            "state": "TASK_STATE_CANCELED",
            "timestamp": now(),
        }
        save_task(task)
    return JSONResponse(public_task(task), media_type=A2A_MEDIA)
