#!/usr/bin/env python3
"""publish: POST a matrix report to a deeprelay benchmark ingest endpoint.

    python3 publish.py report.json --url https://api.deeprelay.ai/internal/benchmarks/ingest \\
        --token-env INGEST_TOKEN

The token is read from the named environment variable and travels only in
the Authorization header. A 5xx or transport failure is retried three times
with backoff (the server failed; resending can help). A 4xx is final: the
report was rejected, its body is printed, and the exit code is non-zero.
On success the server's {run_id, stored} is printed. Stdlib only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ATTEMPTS = 3
BACKOFF_S = 3.0


def publish(body, url, token, opener=urllib.request.urlopen, timeout=60, sleep=time.sleep, log=None):
    """Returns (status, response_text). Raises nothing for HTTP outcomes;
    a transport failure that survives every attempt is re-raised."""
    log = log or (lambda m: None)
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        if attempt > 1:
            sleep(BACKOFF_S * (attempt - 1))
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                     "User-Agent": "deeprelay-model-fingerprint/1"})
        try:
            with opener(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            text = (e.read() or b"").decode("utf-8", "replace") if hasattr(e, "read") else ""
            if e.code >= 500:
                last = (e.code, text)
                log(f"attempt {attempt}: HTTP {e.code}, retrying")
                continue
            return e.code, text
        except Exception as e:  # transport
            last = e
            log(f"attempt {attempt}: {type(e).__name__}: {e}; retrying")
    if isinstance(last, tuple):
        return last
    raise last


def main(argv=None, env=None, opener=urllib.request.urlopen, sleep=time.sleep):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report")
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-env", required=True, help="name of the env var holding the bearer token")
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)
    env = os.environ if env is None else env
    token = env.get(args.token_env) or ""
    if not token:
        print(f"publish: {args.token_env} is not set", file=sys.stderr)
        return 2
    with open(args.report, "rb") as f:
        body = f.read()
    try:
        json.loads(body)
    except ValueError as e:
        print(f"publish: {args.report} is not JSON: {e}", file=sys.stderr)
        return 2
    try:
        status, text = publish(body, args.url, token, opener=opener, timeout=args.timeout, sleep=sleep,
                               log=lambda m: print(f"publish: {m}", file=sys.stderr))
    except Exception as e:
        print(f"publish: transport failure after {ATTEMPTS} attempts: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    if status >= 300:
        print(f"publish: HTTP {status}: {text[:2000]}", file=sys.stderr)
        return 1
    try:
        ack = json.loads(text)
        print(json.dumps({"run_id": ack.get("run_id"), "stored": ack.get("stored")}))
    except ValueError:
        print(text[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
