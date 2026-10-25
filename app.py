import os
import json
import uuid
import hashlib
import hmac
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
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1")
AI_MODEL = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MEDIA = "application/a2a+json"
ALLOWED_ACTIONS = {
    "settle_invoice", "request_approval", "hold_invoice",
    "reject_duplicate", "open_exception",
}
TERMINAL_STATES = {
    "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED", "TASK_STATE_FAILED",
}

# This lock reduces duplicate work in a single instance. PostgreSQL advisory locks
# and row locks provide cross-request coordination in the database.
process_lock = threading.RLock()


def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg.connect(url, row_factory=dict_row)


def init_db():
    with db_connect() as conn:
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
        conn.execute("CREATE INDEX IF NOT EXISTS tasks_principal_idx ON tasks(principal)")
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error(status, code, message):
    return JSONResponse(
        status_code=status,
        media_type=A2A_MEDIA,
        content={"error": {"code": code, "message": message}},
    )


def get_principal(request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    supplied = auth[7:]
    if not supplied or supplied.strip() != supplied:
        return None
    return hashlib.sha256(supplied.encode()).hexdigest()



def check_post_headers(request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "INVALID_VERSION", "A2A-Version must be 1.0")
    if "application/a2a+json" not in request.headers.get("content-type", ""):
        return error(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json")
    return None


def check_version(request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "INVALID_VERSION", "A2A-Version must be 1.0")
    return None


def state_of(task):
    return task.get("status", {}).get("state", "")


def make_message(role, media_type, data, task_id=None, context_id=None):
    result = {
        "messageId": str(uuid.uuid4()),
        "role": role,
        "parts": [{"mediaType": media_type, "data": data}],
    }
    if task_id:
        result["taskId"] = task_id
    if context_id:
        result["contextId"] = context_id
    return result


def find_part_data(message, media_type):
    for part in message.get("parts", []):
        if part.get("mediaType") == media_type:
            return part.get("data")
    return None


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def semantic_package(package):
    # Package IDs can change between Check/Save namespaces; the invoice evidence is the cache key.
    return {k: v for k, v in package.items() if k != "packageId"}


def save_task(conn, task):
    conn.execute(
        """INSERT INTO tasks(id, principal, state, data)
           VALUES (%s, %s, %s, %s::jsonb)
           ON CONFLICT(id) DO UPDATE SET state=EXCLUDED.state,
             data=EXCLUDED.data, updated_at=NOW()""",
        (task["id"], task["_principal"], state_of(task), json.dumps(task)),
    )


def read_task(task_id, principal, conn=None, lock=False):
    own = conn is None
    if own:
        conn = db_connect()
    try:
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            "SELECT data FROM tasks WHERE id=%s AND principal=%s" + suffix,
            (task_id, principal),
        ).fetchone()
        return row["data"] if row else None
    finally:
        if own:
            conn.close()


def public_task(task):
    return {key: value for key, value in task.items() if not key.startswith("_")}


def model_proposals(packages, batch_id, policy_revision):
    if not AI_API_KEY:
        raise RuntimeError("AI_API_KEY is not configured")
    system_prompt = """
You are a careful invoice action proposal agent. Read all invoice packages and return ONLY one JSON object:
{"proposals":[{"packageId":"...","vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR","action":"...","evidenceRefs":["...","...","..."],"rationale":"..."}]}

Return exactly one proposal for every input package, preserving each packageId exactly. Choose exactly one action:
settle_invoice (valid, reconciled, within autonomous authority),
request_approval (commercially valid but outside delegated authority),
hold_invoice (payment pauses until a stated verification completes),
reject_duplicate (the same commercial invoice was already paid),
open_exception (material records conflict).

Use the actual decisive invoice evidence, not merely keyword matching. Documents can contain irrelevant action words,
negation, old examples, archives, and decoys. Copy exactly three decisive bracketed evidence references from the paragraph
that determines the decision. Do not use cover-sheet, archive, or training-decoy references. amountMinor must be integer
minor currency units, currency must be INR, and vendorName/invoiceNumber/amountMinor must be grounded in the evidence.
Rationale must be 60-1500 characters, name the chosen action, and cite at least two selected evidence references.
Never invent missing facts. If records materially conflict, use open_exception and explain the conflict.
""".strip()
    user_payload = {
        "batchId": batch_id,
        "policyRevision": policy_revision,
        "packages": packages,
    }
    response = requests.post(
        AI_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + AI_API_KEY},
        json={
            "model": AI_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=36,
    )
    if not response.ok:
    print("Groq API error status:", response.status_code)
    print("Groq API error body:", response.text[:1000])
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    proposals = parsed.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != len(packages):
        raise ValueError("AI did not return exactly one proposal per package")
    by_id = {}
    for item in proposals:
        pid = item.get("packageId")
        if not pid or pid in by_id:
            raise ValueError("AI returned missing or duplicate package IDs")
        by_id[pid] = item
    expected = {p.get("packageId") for p in packages}
    if set(by_id) != expected:
        raise ValueError("AI package IDs do not match the input batch")
    return by_id


def get_proposals(packages, batch_id, policy_revision):
    # Load cached semantic decisions first; make one model call for all cache misses.
    cached = {}
    misses = []
    with db_connect() as conn:
        for package in packages:
            key = canonical_hash(semantic_package(package))
            row = conn.execute(
                "SELECT proposal FROM decisions WHERE package_hash=%s", (key,)
            ).fetchone()
            if row:
                cached[package["packageId"]] = row["proposal"]
            else:
                misses.append(package)
    model_results = {}
    if misses:
        model_results = model_proposals(misses, batch_id, policy_revision)
        with db_connect() as conn:
            for package in misses:
                result = model_results[package["packageId"]]
                key = canonical_hash(semantic_package(package))
                conn.execute(
                    """INSERT INTO decisions(package_hash, proposal)
                       VALUES (%s, %s::jsonb) ON CONFLICT(package_hash) DO NOTHING""",
                    (key, json.dumps(result)),
                )
                # If another request won the cache race, keep the already stored canonical decision.
                row = conn.execute(
                    "SELECT proposal FROM decisions WHERE package_hash=%s", (key,)
                ).fetchone()
                cached[package["packageId"]] = row["proposal"] if row else result

    proposals = []
    for package in packages:
        source = dict(cached[package["packageId"]])
        action = source.get("action")
        evidence = source.get("evidenceRefs")
        rationale = source.get("rationale", "")
        amount = source.get("amountMinor")
        if action not in ALLOWED_ACTIONS:
            raise ValueError("AI returned an unsupported action")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ValueError("AI returned an invalid amount")
        if source.get("currency") != "INR":
            raise ValueError("AI returned an invalid currency")
        if not isinstance(evidence, list) or len(evidence) != 3 or not all(isinstance(x, str) for x in evidence):
            raise ValueError("AI must return exactly three evidence references")
        if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
            raise ValueError("AI rationale has an invalid length")
        proposals.append({
            "packageId": package["packageId"],
            "actionId": str(uuid.uuid4()) + str(uuid.uuid4()),
            "action": action,
            "facts": {
                "vendorName": source.get("vendorName", ""),
                "invoiceNumber": source.get("invoiceNumber", ""),
                "amountMinor": amount,
                "currency": "INR",
            },
            "evidenceRefs": evidence,
            "rationale": rationale,
        })
    return proposals


def build_initial_task(message, batch, principal):
    batch_id = batch.get("batchId")
    packages = batch.get("packages")
    if not batch_id or not isinstance(packages, list) or not packages:
        raise ValueError("Invalid invoice claim batch")
    if len(packages) > 100:
        raise ValueError("Batch is too large")
    seen = set()
    for package in packages:
        pid = package.get("packageId") if isinstance(package, dict) else None
        if not pid or pid in seen:
            raise ValueError("Package IDs must be present and unique")
        seen.add(pid)
    proposals = get_proposals(packages, batch_id, batch.get("policyRevision", ""))
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    proposal_data = {"batchId": batch_id, "proposals": proposals}
    proposal_message = make_message("ROLE_AGENT", PROPOSAL_MODE, proposal_data, task_id, context_id)
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": utc_now()},
        "history": [message, proposal_message],
        "artifacts": [{
            "artifactId": str(uuid.uuid4()),
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROPOSAL_MODE, "data": proposal_data}],
        }],
        "_principal": principal,
        "_batchId": batch_id,
        "_proposals": proposals,
    }
    return task


def apply_results(task, message, result_data):
    if state_of(task) != "TASK_STATE_INPUT_REQUIRED":
        raise ValueError("Task is not waiting for results")
    if message.get("taskId") != task.get("id"):
        raise ValueError("Task ID does not match")
    if message.get("contextId") != task.get("contextId"):
        raise ValueError("Task context does not match")
    batch_id = task.get("_batchId")
    if result_data.get("batchId") != batch_id:
        raise ValueError("Batch ID does not match")
    proposal_map = {p["packageId"]: p for p in task.get("_proposals", [])}
    results = result_data.get("results")
    if not isinstance(results, list) or len(results) != len(proposal_map):
        raise ValueError("Every proposal must have exactly one result")
    seen = set()
    executions = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Invalid result record")
        pid = result.get("packageId")
        if not pid or pid in seen or pid not in proposal_map:
            raise ValueError("Unknown or repeated package ID")
        seen.add(pid)
        proposal = proposal_map[pid]
        if result.get("actionId") != proposal.get("actionId") or result.get("action") != proposal.get("action"):
            raise ValueError("Result does not match the stored proposal")
        if result.get("outcome") not in ("ACCEPTED", "REJECTED"):
            raise ValueError("Invalid result outcome")
        nonce = result.get("receiptNonce")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("Missing receipt nonce")
        if result["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": pid,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce,
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
            })
    task["history"].append(message)
    task["artifacts"].append({
        "artifactId": str(uuid.uuid4()),
        "name": "invoice-action-receipts",
        "parts": [{"mediaType": RECEIPT_MODE,
                   "data": {"batchId": batch_id, "executions": executions}}],
    })
    task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": utc_now()}
    return task


@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "name": "A2A Invoice Action Agent",
        "description": "Proposes evidence-based invoice actions and records accepted results.",
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
    problem = check_post_headers(request)
    if problem:
        return problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    try:
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
            return error(400, "INVALID_MESSAGE", "Message is required")
        message = body["message"]
        mid = message.get("messageId")
        if not isinstance(mid, str) or not mid:
            return error(400, "INVALID_MESSAGE", "messageId is required")
        digest = canonical_hash(message)  # configuration is deliberately excluded
        lock_key = int.from_bytes(hashlib.sha256((principal + mid).encode()).digest()[:8], "big", signed=True)

        with process_lock, db_connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            existing = conn.execute(
                "SELECT body_hash, task_id FROM idempotency WHERE principal=%s AND message_id=%s",
                (principal, mid),
            ).fetchone()
            if existing:
                if existing["body_hash"] != digest:
                    return error(409, "IDEMPOTENCY_CONFLICT", "messageId was used with different content")
                task = read_task(existing["task_id"], principal, conn=conn)
                if task:
                    return JSONResponse({"task": public_task(task)}, media_type=A2A_MEDIA)
                return error(500, "STATE_UNAVAILABLE", "Stored task is unavailable")

            results_data = find_part_data(message, RESULT_MODE)
            if results_data is not None:
                task_id = message.get("taskId")
                if not task_id:
                    return error(400, "INVALID_MESSAGE", "taskId is required for results")
                task = read_task(task_id, principal, conn=conn, lock=True)
                if not task:
                    return error(404, "TASK_NOT_FOUND", "Task not found")
                if state_of(task) in TERMINAL_STATES:
                    return error(409, "TASK_TERMINAL", "Task is already terminal")
                try:
                    task = apply_results(task, message, results_data)
                except ValueError:
                    return error(400, "INVALID_RESULTS", "Results do not match the stored task")
            else:
                batch = find_part_data(message, INPUT_MODE)
                if not isinstance(batch, dict):
                    return error(400, "INVALID_MESSAGE", "Invoice claim batch part is required")
                try:
                    task = build_initial_task(message, batch, principal)
                except (ValueError, KeyError, requests.RequestException, RuntimeError, json.JSONDecodeError):
                    return error(502, "PROCESSING_FAILED", "Could not safely process this batch")

            task["_principal"] = principal
            save_task(conn, task)
            conn.execute(
                "INSERT INTO idempotency(principal,message_id,body_hash,task_id) VALUES (%s,%s,%s,%s)",
                (principal, mid, digest, task["id"]),
            )
            return JSONResponse({"task": public_task(task)}, media_type=A2A_MEDIA)
    except Exception:
        return error(500, "INTERNAL_ERROR", "Request could not be processed")


@app.get("/tasks")
def list_tasks(request: Request):
    problem = check_version(request)
    if problem:
        return problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT data FROM tasks WHERE principal=%s ORDER BY created_at DESC", (principal,)
        ).fetchall()
    return JSONResponse({"tasks": [public_task(row["data"]) for row in rows]}, media_type=A2A_MEDIA)


@app.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    problem = check_version(request)
    if problem:
        return problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    task = read_task(task_id, principal)
    if not task:
        return error(404, "TASK_NOT_FOUND", "Task not found")
    return JSONResponse(public_task(task), media_type=A2A_MEDIA)


@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    problem = check_post_headers(request)
    if problem:
        return problem
    principal = get_principal(request)
    if not principal:
        return error(401, "UNAUTHORIZED", "Valid bearer token required")
    with process_lock, db_connect() as conn:
        task = read_task(task_id, principal, conn=conn, lock=True)
        if not task:
            return error(404, "TASK_NOT_FOUND", "Task not found")
        if state_of(task) in TERMINAL_STATES:
            return error(409, "TASK_TERMINAL", "Task is already terminal")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": utc_now()}
        save_task(conn, task)
    return JSONResponse(public_task(task), media_type=A2A_MEDIA)
