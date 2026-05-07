"""
Custom Vast.ai PyWorker for ComfyUI
------------------------------------
Ported from RunPod handler.py — direct WebSocket calls to ComfyUI,
no api-wrapper queue overhead, S3 upload to R2.

Repo structure expected by start_server.sh:
    workers/
        comfyui-json/
            worker.py       ← this file
    requirements.txt

Environment variables:
    COMFY_HOST              ComfyUI host:port (default: 127.0.0.1:18188)
    S3_BUCKET_NAME          R2 bucket name
    S3_ACCESS_KEY_ID        R2 access key
    S3_SECRET_ACCESS_KEY    R2 secret key
    S3_ENDPOINT_URL         R2 endpoint (https://ACCOUNT.r2.cloudflarestorage.com)
    S3_REGION               R2 region (default: auto)
    PRESIGN_EXPIRY          Presigned URL expiry in seconds (default: 3600)
"""

import asyncio
import base64
import json
import logging
import os
import traceback
import uuid
from io import BytesIO

import aiohttp
import boto3
from botocore.exceptions import ClientError

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("comfyui-worker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMFY_HOST      = os.getenv("COMFY_HOST", "127.0.0.1:18188")
COMFY_HTTP      = f"http://{COMFY_HOST}"
COMFY_WS        = f"ws://{COMFY_HOST}"
MODEL_LOG_FILE  = os.getenv("MODEL_LOG", "/var/log/portal/comfyui.log")

S3_BUCKET       = os.getenv("S3_BUCKET_NAME")
S3_KEY          = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET       = os.getenv("S3_SECRET_ACCESS_KEY")
S3_ENDPOINT     = os.getenv("S3_ENDPOINT_URL")
S3_REGION       = os.getenv("S3_REGION", "auto")
PRESIGN_EXPIRY  = int(os.getenv("PRESIGN_EXPIRY", "3600"))

WEBSOCKET_RECONNECT_ATTEMPTS = int(os.getenv("WEBSOCKET_RECONNECT_ATTEMPTS", "5"))
WEBSOCKET_RECONNECT_DELAY    = int(os.getenv("WEBSOCKET_RECONNECT_DELAY_S", "3"))

# ---------------------------------------------------------------------------
# S3 / R2 client
# ---------------------------------------------------------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        region_name=S3_REGION,
    )


def upload_to_r2(image_bytes: bytes, filename: str) -> str:
    """Upload image bytes to R2 and return a presigned URL."""
    s3 = get_s3_client()
    key = f"outputs/{uuid.uuid4()}/{filename}"
    s3.upload_fileobj(BytesIO(image_bytes), S3_BUCKET, key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    return url


# ---------------------------------------------------------------------------
# ComfyUI helpers
# ---------------------------------------------------------------------------
async def wait_for_comfy(session: aiohttp.ClientSession, retries=100, delay=0.5):
    """Poll ComfyUI until it responds on /."""
    for _ in range(retries):
        try:
            async with session.get(f"{COMFY_HTTP}/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(delay)
    return False


async def upload_input_images(session: aiohttp.ClientSession, images: list):
    """Upload base64 input images to ComfyUI /upload/image."""
    errors = []
    for img in images:
        try:
            name = img["name"]
            data_uri = img["image"]
            # Strip data URI prefix if present
            b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
            blob = base64.b64decode(b64)
            form = aiohttp.FormData()
            form.add_field("image", BytesIO(blob), filename=name, content_type="image/png")
            form.add_field("overwrite", "true")
            async with session.post(
                f"{COMFY_HTTP}/upload/image",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                log.info("Uploaded input image: %s", name)
        except Exception as e:
            errors.append(f"Failed to upload {img.get('name', '?')}: {e}")
    return errors


async def queue_workflow(session: aiohttp.ClientSession, workflow: dict, client_id: str) -> str:
    """Submit workflow to ComfyUI and return prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    async with session.post(
        f"{COMFY_HTTP}/prompt",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        if resp.status == 400:
            body = await resp.text()
            raise ValueError(f"ComfyUI workflow validation failed: {body}")
        resp.raise_for_status()
        data = await resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise ValueError(f"Missing prompt_id in response: {data}")
    return prompt_id


async def wait_for_execution(client_id: str, prompt_id: str) -> None:
    """Connect to ComfyUI WebSocket and wait for execution to finish."""
    ws_url = f"{COMFY_WS}/ws?clientId={client_id}"
    attempts = 0

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=10)) as ws:
                    log.info("WebSocket connected for prompt %s", prompt_id)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "executing":
                                node_data = data.get("data", {})
                                if (
                                    node_data.get("node") is None
                                    and node_data.get("prompt_id") == prompt_id
                                ):
                                    log.info("Execution complete for %s", prompt_id)
                                    return
                            elif data.get("type") == "execution_error":
                                err = data.get("data", {})
                                raise RuntimeError(
                                    f"ComfyUI execution error — node {err.get('node_id')} "
                                    f"({err.get('node_type')}): {err.get('exception_message')}"
                                )
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            raise aiohttp.ClientConnectionError("WebSocket closed unexpectedly")
            return  # clean exit

        except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as e:
            attempts += 1
            if attempts >= WEBSOCKET_RECONNECT_ATTEMPTS:
                raise RuntimeError(f"WebSocket reconnect failed after {attempts} attempts: {e}")
            log.warning("WebSocket disconnected, retrying (%d/%d)...", attempts, WEBSOCKET_RECONNECT_ATTEMPTS)
            await asyncio.sleep(WEBSOCKET_RECONNECT_DELAY)


async def get_outputs(session: aiohttp.ClientSession, prompt_id: str) -> dict:
    """Fetch history and return output nodes."""
    async with session.get(
        f"{COMFY_HTTP}/history/{prompt_id}",
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp.raise_for_status()
        history = await resp.json()
    if prompt_id not in history:
        raise ValueError(f"prompt_id {prompt_id} not found in history")
    return history[prompt_id].get("outputs", {})


async def fetch_image(session: aiohttp.ClientSession, filename: str, subfolder: str, img_type: str) -> bytes | None:
    """Fetch image bytes from ComfyUI /view endpoint."""
    params = {"filename": filename, "subfolder": subfolder, "type": img_type}
    try:
        async with session.get(
            f"{COMFY_HTTP}/view",
            params=params,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            return await resp.read()
    except Exception as e:
        log.error("Failed to fetch image %s: %s", filename, e)
        return None


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
async def handler(**payload) -> dict:
    """
    Main generation handler.

    Expected input:
        {
            "input": {
                "workflow_json": { ... ComfyUI API workflow ... },
                "images": [                          # optional input images
                    {"name": "input.png", "image": "<base64>"}
                ]
            }
        }

    Returns:
        {
            "images": [
                {"filename": "output.png", "type": "url",    "data": "https://..."},
                {"filename": "output.png", "type": "base64", "data": "<base64>"}
            ],
            "errors": [...]   # only present if non-fatal warnings occurred
        }
    """
    # SDK passes payload as kwargs — reconstruct dict and extract "input"
    payload_dict = dict(payload)
    job_input = payload_dict.get("input", payload_dict)

    # Support both "workflow_json" (Vast style) and "workflow" (RunPod style)
    workflow = job_input.get("workflow_json") or job_input.get("workflow")
    if not workflow:
        return {"error": "Missing 'workflow_json' in input"}

    input_images = job_input.get("images") or []
    client_id    = str(uuid.uuid4())
    output_data  = []
    errors       = []

    async with aiohttp.ClientSession() as session:

        # 1. Wait for ComfyUI to be ready
        if not await wait_for_comfy(session):
            return {"error": f"ComfyUI at {COMFY_HTTP} not reachable"}

        # 2. Upload any input images
        if input_images:
            upload_errors = await upload_input_images(session, input_images)
            if upload_errors:
                return {"error": "Failed to upload input images", "details": upload_errors}

        # 3. Queue workflow
        try:
            prompt_id = await queue_workflow(session, workflow, client_id)
            log.info("Queued workflow: %s", prompt_id)
        except ValueError as e:
            return {"error": str(e)}

        # 4. Wait for completion via WebSocket
        try:
            await wait_for_execution(client_id, prompt_id)
        except RuntimeError as e:
            return {"error": str(e)}

        # 5. Fetch outputs from history
        try:
            outputs = await get_outputs(session, prompt_id)
        except ValueError as e:
            return {"error": str(e)}

        # 6. Collect and upload/encode output images
        for node_id, node_output in outputs.items():
            # Handle both images and videos (gifs key)
            for output_key in ("images", "gifs"):
                for item in node_output.get(output_key, []):
                    filename  = item.get("filename")
                    subfolder = item.get("subfolder", "")
                    item_type = item.get("type")

                    if item_type == "temp":
                        continue
                    if not filename:
                        errors.append(f"Node {node_id}: missing filename in output")
                        continue

                    image_bytes = await fetch_image(session, filename, subfolder, item_type)
                    if not image_bytes:
                        errors.append(f"Failed to fetch output file: {filename}")
                        continue

                    # Upload to R2 if configured, otherwise return base64
                    if S3_BUCKET and S3_KEY and S3_SECRET and S3_ENDPOINT:
                        try:
                            url = await asyncio.get_event_loop().run_in_executor(
                                None, upload_to_r2, image_bytes, filename
                            )
                            output_data.append({
                                "filename": filename,
                                "type": "url",
                                "data": url,
                            })
                            log.info("Uploaded %s to R2", filename)
                        except Exception as e:
                            errors.append(f"R2 upload failed for {filename}: {e}")
                    else:
                        b64 = base64.b64encode(image_bytes).decode("utf-8")
                        output_data.append({
                            "filename": filename,
                            "type": "base64",
                            "data": b64,
                        })
                        log.info("Returning %s as base64", filename)

    if not output_data and errors:
        return {"error": "Job failed with no output", "details": errors}

    result: dict = {"images": output_data}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# Ping benchmark — required by vastai SDK but does no real work
# No images generated, no S3 uploads, just a fast no-op to satisfy the SDK
# ---------------------------------------------------------------------------
import time

def ping_generator() -> dict:
    return {"ping": True, "ts": time.time()}

async def ping_remote(**params):
    return {"ok": True}

# ---------------------------------------------------------------------------
# Vast Worker config
# ---------------------------------------------------------------------------
worker_config = WorkerConfig(
    model_server_url=f"http://{COMFY_HOST.split(':')[0]}",
    model_server_port=int(COMFY_HOST.split(":")[-1]) if ":" in COMFY_HOST else 18188,
    model_log_file=MODEL_LOG_FILE,

    model_healthcheck_url=f"http://{COMFY_HOST}/system_stats",
    
    handlers=[
        # Lightweight ping benchmark — satisfies SDK requirement, no junk images
        HandlerConfig(
            route="/benchmark/ping",
            allow_parallel_requests=True,
            max_queue_time=0.0,
            benchmark_config=BenchmarkConfig(
                generator=ping_generator,
                runs=1,
                concurrency=1,
                do_warmup=False,
            ),
            remote_function=ping_remote,
        ),
        # Main generation handler
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,   # ComfyUI is single-queue
            max_queue_time=0.0,              # Reject immediately if busy → Vast re-routes
            workload_calculator=lambda _: 100.0,
            benchmark_config=None,
            remote_function=handler,
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=["To see the GUI go to: "],
        on_error=[
            "[ERROR] Provisioning Script failed",
            "PRESTARTUP FAILED",
            "No space left on device",
        ],
        on_info=[],
    ),
)

Worker(worker_config).run()
