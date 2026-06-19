#!/usr/bin/env python3
"""MinerU document-to-markdown converter CLI.

Converts documents to Markdown using MinerU Precision Extract API
(POST /api/v4/extract/task).

Usage:
    python mineru_doc2md.py <file_or_url> [options]
"""

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# Error code table (from reference/error_code.md)
# ---------------------------------------------------------------------------
ERROR_HINTS = {
    "A0202": "Invalid Token — check Bearer prefix or replace with a new Token",
    "A0211": "Token expired — replace with a new Token",
    "-500":  "Parameter error — check parameter types and Content-Type",
    "-10001": "Service error — please try again later",
    "-10002": "Request parameter error — check request parameter format",
    "-60001": "Failed to generate upload URL — please try again later",
    "-60002": "Failed to match file format — ensure filename has a supported extension (pdf/doc/docx/ppt/pptx/xls/xlsx/png/jpg/jpeg/jp2/webp/gif/bmp)",
    "-60003": "File read failure — check if the file is corrupted",
    "-60004": "Empty file — please upload a valid file",
    "-60005": "File size exceeds limit (max 200 MB)",
    "-60006": "Page count exceeds limit (max 200) — please split the file",
    "-60007": "Model service temporarily unavailable — try again later",
    "-60008": "File read timeout — check that the URL is accessible",
    "-60009": "Task submission queue is full — try again later",
    "-60010": "Extract failed — please try again later",
    "-60011": "Failed to get valid file — ensure the file has been uploaded",
    "-60012": "Task not found — ensure task_id is valid",
    "-60013": "No permission to access this task",
    "-60015": "File conversion failed — try manually converting to PDF",
    "-60017": "Retry limit reached",
    "-60018": "Daily extract task limit reached — try again tomorrow",
    "-60019": "HTML file extract quota exhausted — try again tomorrow",
    "-60022": "Web page read failure — network issues or rate limiting",
}

BASE_URL = "https://mineru.net"
POLL_INTERVAL = 10   # seconds between status polls
POLL_MAX = 60        # max attempts → ~10 minutes total


# ---------------------------------------------------------------------------
# .env token loading
# ---------------------------------------------------------------------------

def load_token(env_path: Path) -> str:
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("MINERU_TOKEN="):
            value = line[len("MINERU_TOKEN="):].strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value
    return ""


def require_token(env_path: Path) -> str:
    token = load_token(env_path)
    if not token:
        print(
            "\n[MinerU] MINERU_TOKEN not found.\n"
            "\nPlease:\n"
            "  1. Get your token at https://mineru.net/apiManage/token\n"
            f"  2. Add it to {env_path}:\n\n"
            "       MINERU_TOKEN=your_token_here\n\n"
            "  3. Re-run this command.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no extra dependencies)
# ---------------------------------------------------------------------------

def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _check_api_response(resp_json: dict, context: str = "") -> None:
    code = resp_json.get("code", -1)
    if code == 0:
        return
    msg = resp_json.get("msg", "unknown error")
    trace = resp_json.get("trace_id", "")
    hint = ERROR_HINTS.get(str(code), "")
    lines = [f"\n[MinerU] API error {code}: {msg}"]
    if hint:
        lines.append(f"  Hint: {hint}")
    if trace:
        lines.append(f"  trace_id: {trace}")
    if context:
        lines.append(f"  Context: {context}")
    print("\n".join(lines), file=sys.stderr)
    sys.exit(1)


def _post_json(url: str, token: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers=_auth_headers(token), method="POST")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body_text = e.read().decode(errors="replace")
        print(f"\n[MinerU] HTTP {e.code} at {url}:\n{body_text}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"\n[MinerU] Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _get_json(url: str, token: str) -> dict:
    headers = {k: v for k, v in _auth_headers(token).items() if k != "Content-Type"}
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body_text = e.read().decode(errors="replace")
        print(f"\n[MinerU] HTTP {e.code} at {url}:\n{body_text}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"\n[MinerU] Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _download_bytes(url: str) -> bytes:
    try:
        with urlopen(url) as resp:
            return resp.read()
    except (HTTPError, URLError) as e:
        print(f"\n[MinerU] Download failed: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def _extract_markdown_from_zip(zip_bytes: bytes) -> str:
    """Extract full.md from the MinerU result ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.endswith("full.md")]
        if not candidates:
            print(
                f"\n[MinerU] 'full.md' not found in ZIP.\nFiles present: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        target = sorted(candidates, key=len)[0]
        return zf.read(target).decode("utf-8")


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

def _poll_task(task_id: str, token: str) -> dict:
    """Poll GET /api/v4/extract/task/{task_id} until done or failed."""
    status_url = f"{BASE_URL}/api/v4/extract/task/{task_id}"
    state = "pending"
    for i in range(1, POLL_MAX + 1):
        resp = _get_json(status_url, token)
        _check_api_response(resp, context=f"polling task {task_id}")
        data = resp["data"]
        state = data.get("state", "")
        progress = data.get("extract_progress", {})
        done_p = progress.get("extracted_pages", "?")
        total_p = progress.get("total_pages", "?")
        print(f"  [{i}/{POLL_MAX}] state={state}  pages={done_p}/{total_p}")

        if state == "done":
            return data
        if state == "failed":
            err = data.get("err_msg", "no details")
            print(f"\n[MinerU] Task failed: {err}", file=sys.stderr)
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    print(
        f"\n[MinerU] Timed out after {POLL_MAX * POLL_INTERVAL}s. Last state: {state}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def _build_task_body(args: argparse.Namespace, url: str) -> dict:
    """Build the POST /api/v4/extract/task request body from CLI args."""
    body: dict = {"url": url}

    if args.model_version:
        body["model_version"] = args.model_version
    if args.is_ocr:
        body["is_ocr"] = True
    if not args.enable_formula:
        body["enable_formula"] = False
    if not args.enable_table:
        body["enable_table"] = False
    if args.language:
        body["language"] = args.language
    if args.page_ranges:
        body["page_ranges"] = args.page_ranges
    if args.extra_formats:
        body["extra_formats"] = args.extra_formats
    if args.no_cache:
        body["no_cache"] = True
    if args.cache_tolerance is not None:
        body["cache_tolerance"] = args.cache_tolerance
    if args.data_id:
        body["data_id"] = args.data_id

    return body


def _upload_local_file(file_path: Path, token: str, args: argparse.Namespace) -> str:
    """Upload a local file to MinerU OSS and return the batch_id.

    Extraction options are declared here (at presigned-URL generation time).
    Uploading to the presigned URL triggers extraction automatically —
    no separate batch-submit step is needed.
    """
    print(f"[MinerU] Getting presigned upload URL for: {file_path.name}")
    file_opts: dict = {"name": file_path.name, "data_id": "cli-upload"}
    if args.model_version:
        file_opts["model_version"] = args.model_version
    if args.is_ocr:
        file_opts["is_ocr"] = True
    if not args.enable_formula:
        file_opts["enable_formula"] = False
    if not args.enable_table:
        file_opts["enable_table"] = False
    if args.language:
        file_opts["language"] = args.language
    if args.page_ranges:
        file_opts["page_ranges"] = args.page_ranges
    if args.extra_formats:
        file_opts["extra_formats"] = args.extra_formats

    resp = _post_json(
        f"{BASE_URL}/api/v4/file-urls/batch",
        token,
        {"files": [file_opts]},
    )
    _check_api_response(resp, context="get presigned URL")

    batch_id: str = resp["data"]["batch_id"]
    presigned_url: str = resp["data"]["file_urls"][0]

    size_kb = file_path.stat().st_size / 1024
    print(f"[MinerU] Uploading {size_kb:.1f} KB...")
    # Use -w to append the HTTP status code after the response body, then check it.
    # curl exits 0 even on HTTP 4xx, so we must verify the status explicitly.
    result = subprocess.run(
        ["curl", "--progress-bar", "-X", "PUT", "--data-binary", f"@{file_path}",
         "-H", "Content-Type:", "-w", "\n%{http_code}", presigned_url],
        stderr=sys.stderr,
        stdout=subprocess.PIPE,
        text=True,
    )
    http_code = result.stdout.strip().split("\n")[-1].strip()
    if result.returncode != 0 or not http_code.startswith("2"):
        print(f"\n[MinerU] Upload failed (curl exit {result.returncode}, HTTP {http_code})", file=sys.stderr)
        sys.exit(1)
    print("[MinerU] Upload complete.")

    return batch_id


def _poll_batch(batch_id: str, token: str) -> str:
    """Poll GET /api/v4/extract-results/batch/{batch_id} until done, return full_zip_url."""
    status_url = f"{BASE_URL}/api/v4/extract-results/batch/{batch_id}"
    state = "pending"
    for i in range(1, POLL_MAX + 1):
        resp = _get_json(status_url, token)
        _check_api_response(resp, context=f"polling batch {batch_id}")
        results = resp["data"].get("extract_result") or resp["data"].get("files") or []
        if results:
            first = results[0]
            state = first.get("state", "")
            progress = first.get("extract_progress", {})
            done_p = progress.get("extracted_pages", "?")
            total_p = progress.get("total_pages", "?")
            print(f"  [{i}/{POLL_MAX}] state={state}  pages={done_p}/{total_p}")
            if state == "done":
                zip_url = first.get("full_zip_url", "")
                if not zip_url:
                    print(f"\n[MinerU] No full_zip_url in result:\n{json.dumps(first, indent=2)}", file=sys.stderr)
                    sys.exit(1)
                return zip_url
            if state == "failed":
                print(f"\n[MinerU] Task failed: {first.get('err_msg', 'no details')}", file=sys.stderr)
                sys.exit(1)
        else:
            state = resp["data"].get("state", "pending")
            print(f"  [{i}/{POLL_MAX}] batch state={state}")
        time.sleep(POLL_INTERVAL)

    print(f"\n[MinerU] Timed out after {POLL_MAX * POLL_INTERVAL}s. Last state: {state}", file=sys.stderr)
    sys.exit(1)


def convert(args: argparse.Namespace, token: str) -> str:
    """Run extraction and return markdown content."""
    input_val: str = args.input

    if _is_url(input_val):
        print(f"[MinerU] Submitting URL: {input_val}")
        body = _build_task_body(args, input_val)
        resp = _post_json(f"{BASE_URL}/api/v4/extract/task", token, body)
        _check_api_response(resp, context="submit task")
        task_id: str = resp["data"]["task_id"]
        print(f"[MinerU] task_id={task_id}  Polling...")
        data = _poll_task(task_id, token)
        zip_url: str = data.get("full_zip_url", "")
        if not zip_url:
            print(f"\n[MinerU] No full_zip_url in response:\n{json.dumps(data, indent=2)}", file=sys.stderr)
            sys.exit(1)
    else:
        file_path = Path(input_val)
        if not file_path.exists():
            print(f"\n[MinerU] File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        batch_id = _upload_local_file(file_path, token, args)
        print(f"[MinerU] batch_id={batch_id}  Polling...")
        zip_url = _poll_batch(batch_id, token)

    print("[MinerU] Downloading result ZIP...")
    zip_bytes = _download_bytes(zip_url)
    print("[MinerU] Extracting full.md from ZIP...")
    return _extract_markdown_from_zip(zip_bytes)


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def _is_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def _resolve_output(input_val: str, output_arg: Optional[str]) -> Path:
    if output_arg:
        return Path(output_arg)
    if _is_url(input_val):
        stem = Path(urlparse(input_val).path).stem or "output"
        return Path.cwd() / f"{stem}.md"
    return Path(input_val).with_suffix(".md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mineru_doc2md",
        description="Convert documents to Markdown via MinerU Precision Extract API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s paper.pdf
  %(prog)s paper.pdf -o notes/paper.md
  %(prog)s https://example.com/paper.pdf --language en --page-ranges 1-10
  %(prog)s paper.pdf --model-version vlm --is-ocr --extra-formats docx html
        """,
    )

    parser.add_argument("input", help="Local file path or document URL")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="Output .md file path (default: same dir/name as input with .md suffix)")

    api = parser.add_argument_group("API parameters")
    api.add_argument(
        "--model-version",
        choices=["pipeline", "vlm", "MinerU-HTML"],
        default=None,
        help="Parsing model (default: pipeline). 'vlm' for vision-language model.",
    )
    api.add_argument("--is-ocr", action="store_true", default=False,
                     help="Enable OCR. Useful for scanned documents.")
    api.add_argument("--no-formula", dest="enable_formula", action="store_false", default=True,
                     help="Disable formula recognition (enabled by default).")
    api.add_argument("--no-table", dest="enable_table", action="store_false", default=True,
                     help="Disable table recognition (enabled by default).")
    api.add_argument("--language", metavar="LANG", default=None,
                     help="Document language code (e.g. ch, en, ja). Default: ch.")
    api.add_argument("--page-ranges", metavar="RANGES", default=None,
                     help='Pages to parse, e.g. "1,3-5".')
    api.add_argument("--extra-formats", nargs="+", choices=["docx", "html", "latex"],
                     metavar="FMT", default=None,
                     help="Additional export formats in result ZIP.")
    api.add_argument("--no-cache", action="store_true", default=False,
                     help="Bypass URL content caching.")
    api.add_argument("--cache-tolerance", type=int, metavar="SECS", default=None,
                     help="Cache validity window in seconds (default: 900).")
    api.add_argument("--data-id", metavar="ID", default=None,
                     help="Custom identifier (max 128 chars) for business tracking.")
    parser.add_argument("--env", metavar="FILE", default=None,
                        help="Path to .env file (default: .env in current directory).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    env_path = Path(args.env) if args.env else Path.cwd() / ".env"
    token = require_token(env_path)

    content = convert(args, token)

    out_path = _resolve_output(args.input, args.output)
    if out_path.exists():
        answer = input(f"\n[MinerU] '{out_path}' already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("[MinerU] Aborted.")
            sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n[MinerU] Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
