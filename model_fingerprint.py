#!/usr/bin/env python3
"""model_fingerprint: model identity check by top-k logprob fingerprinting.

Answers "is the deployment serving deeprelay/<model> really that model, and
not a cheaper, quantized or substituted build?" Stdlib only, so anyone with
their own keys can rerun it. The method is written up in METHOD.md.

  * 12 fixed prompts at temperature 0, seed 42, logprobs with top_logprobs.
  * Lanes (matrix.json):
      subject   - the deeprelay model id through api.deeprelay.ai (judged);
      reference - an independently operated deployment of the same weights
                  (two are needed: their mutual distance is the same-model
                  band);
      control   - a model that is deliberately NOT the target, which must
                  separate from the references or the metrics have not been
                  shown to discriminate;
      relay     - accepted as an alias of subject.
  * prompt_tokens across the 12 prompts fingerprints the tokenizer plus the
    chat template. It is compared as a RELATIVE shape (each count minus the
    lane's own count on the anchor prompt) so a provider that prepends a
    fixed system prompt still agrees, while a different tokenizer still
    shows up in the per-prompt deltas. The signature can prove a mismatch
    but never a match on its own.
  * Only CONTENT tokens are compared. A reasoning model's logprob stream may
    cover reasoning + content on one provider and content only on another;
    each response is sliced to the suffix that spells message.content before
    anything is measured.
  * Jensen-Shannon distance over the renormalized top-k distributions,
    averaged over the common greedy prefix, fingerprints the weights plus
    serving precision. Widths differ across lanes, so every comparison is
    truncated to the common k and a lane is judged at its own k against a
    reference band computed at that same k. k = 1 (the chosen token only)
    uses the chosen-token |delta logprob| instead.
  * A lane passes when it is within JS_TOLERANCE x the larger of the
    reference band and the replicate noise floor, where noise may raise the
    limit by at most NOISE_CAP_X x the band, under an absolute ceiling.
    A mismatch needs decisive evidence: the signature, or a distribution
    distance at width >= 2 over the limit. Top-1 evidence (width 1) can
    confirm a match but never fail a lane on its own: over the limit it
    is inconclusive.
  * Prompts whose greedy paths diverge at the first content token carry no
    distance: counted as unaligned, never scored 1.0.
  * Negative controls must separate, judged against the LARGEST limit any
    verdict lane received, by the signature or by every metric a verdict
    relied on. Otherwise the run is INCONCLUSIVE, never a pass.

Guards (a run that verified nothing must never read as green):

  * a lane whose key env var is unset is SKIPPED, loudly;
  * zero subject lanes is INSUFFICIENT;
  * a 429, 5xx or transport error is retried MAX_RETRIES times with backoff;
    a prompt counts toward coverage if ANY replicate came back;
  * every subject lane must return at least MIN_COMPARED of the 12 prompts,
    and at least two references must return top-k logprobs on MIN_COMPARED
    prompts, else INSUFFICIENT;
  * a lane that asked for logprobs and got none is at best INCONCLUSIVE.

Verdicts per subject lane: same_model, mismatch, inconclusive.
Overall: pass, mismatch, inconclusive, insufficient, error. The single-model
CLI exits 0 only on pass; run_matrix.py has its own exit contract.

Library use (run() never raises; a crash becomes verdict "error"):

    import model_fingerprint as mf
    report = mf.run({"model": "deeprelay/deepseek-v4.1-flash"},
                    env={"DEEPRELAY_API_KEY": ..., "TOGETHER_API_KEY": ...,
                         "FIREWORKS_API_KEY": ...})
    report["verdict"]

CLI:

    export DEEPRELAY_API_KEY=... TOGETHER_API_KEY=... FIREWORKS_API_KEY=...
    python3 model_fingerprint.py --model deeprelay/deepseek-v4.1-flash \\
        --json-out fingerprint.json
    python3 model_fingerprint.py --analyze-only      # re-read the latest run
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HARNESS_VERSION = "2026.11.04"

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_LANES_FILE = HERE / "matrix.json"
DEFAULT_MODEL = "deeprelay/deepseek-v4.1-flash"
DEFAULT_RAW_DIR = pathlib.Path(tempfile.gettempdir()) / "deeprelay-fingerprint"
DEFAULT_REPLICATES = 2
DEFAULT_MAX_REQUESTS = 300
DEFAULT_MAX_TOKENS = 300
DEFAULT_TIMEOUT = 180
DEFAULT_WORKERS = 4
# Whole-run budget per model. Thinking-on models (max_tokens 2000) need the
# headroom; the AWS probe runner passes its own --deadline-s.
DEFAULT_DEADLINE_S = 480
# Widest top-k any lane is judged at. Providers honour at most 5 and the
# same-model band was calibrated at 5.
TOPK_CAP = 5

# A lane passes the distribution test when its mean distance to the
# references is within JS_TOLERANCE x max(reference band, noise floor),
# where the noise floor (the largest replicate noise among the references
# and the lane) may raise the limit by at most NOISE_CAP_X x the band: a
# noisy substitute must not buy itself a wide limit with its own jitter.
JS_TOLERANCE = 1.5
NOISE_CAP_X = 2.0
# Absolute ceilings, so a drifted reference cannot widen the tolerance until
# anything passes. Calibration (2026-10-12, 2026-10-14, 2026-11-03 runs):
# JS band 0.019-0.057 at k=5, chosen-token |dlogprob| band 0.011-0.055.
JS_LIMIT_CEILING = 0.20
LP_LIMIT_CEILING = 0.20
JS_BAND_CEILING = round(JS_LIMIT_CEILING / JS_TOLERANCE, 4)
LP_BAND_CEILING = round(LP_LIMIT_CEILING / JS_TOLERANCE, 4)
# Retries per probe on 429, 5xx or a transport error, with linear backoff
# (RETRY_BACKOFF_S, 2 x RETRY_BACKOFF_S, ...). Together rate-limits some
# models dynamically, so one retry is not enough.
MAX_RETRIES = 2
RETRY_BACKOFF_S = 2.0
RETRYABLE = ("rate_limited", "http_5xx", "timeout", "connect_error", "malformed")
# Coverage floors. A verdict lane (and each reference) must return at least
# MIN_COMPARED of the 12 prompts, and a distance comparison needs at least
# MIN_ALIGNED prompts whose greedy content paths share a prefix.
MIN_COMPARED = 10
MIN_ALIGNED = 6
# A control separates on the signature when at least this fraction of the
# comparable prompts disagree on the relative prompt_tokens shape: 2 of 12.
# A subject is a mismatch on ONE disagreeing prompt; a control gets one
# prompt of slack against a flaky count. The old 0.5 was calibrated for
# absolute counts, where a different tokenizer differed on every prompt;
# relative shapes absorb the constant and two English-efficient BPE
# tokenizers agree on most English prompts, differing on the CJK, Cyrillic
# and tokenizer-stress prompts (measured 4/12 for gpt-oss vs DeepSeek).
CONTROL_SIGNATURE_SEPARATION = 1 / 6
# A reference the subject reproduces with identical content on every
# prompt, at a distance no larger than DEPENDENT_NOISE_X times that
# reference's own replicate noise, or under
# DEPENDENT_DISTANCE outright, is the subject's own upstream, not an
# independent deployment. Independent deployments of the same weights
# measure 0.006-0.05 apart; a relay to the same deployment measures its
# replicate noise (0-0.002).
DEPENDENT_DISTANCE = 1e-4
DEPENDENT_NOISE_X = 1.5
# ... but never above this: a noisy reference must not widen "the same
# deployment" until an independent one qualifies. Measured dependent
# cases: 4e-8 to 1.6e-3; independent same-weights pairs: 0.006 and up.
DEPENDENT_CEILING = 0.004
# Tokens a provider may emit between reasoning and content that are not
# part of message.content (a closing think tag, blank lines).
BOUNDARY_TOKENS = ("</think>", "<|end_of_thinking|>", "<|im_end|>")

# Fixed probe set. Short deterministic answers keep cost down; the tokenizer
# and multilingual items exist to catch a different model family outright.
# Escapes keep the invisible characters (NBSP, ZWSP) visible in review.
PROMPTS = {
    "p01_capital": "What is the capital of France? One word.",
    "p02_arith": "Compute 17 * 23. Answer with the number only.",
    "p03_tokenizer": ("Repeat this string exactly, nothing else: "
                      "```indivisible\xa0\U0001f525SolidGoldMagikarp​～```"),
    "p04_chinese": "用一句话解释什么是熵。",
    "p05_code": "Write a Python one-liner that reverses a string. Code only.",
    "p06_memorize": "Complete the next three words exactly: 'It was the best of times, it was'",
    "p07_json": 'Output this and nothing else: {"a":1}',
    "p08_logic": ("If all bloops are razzies and all razzies are lazzies, "
                  "are all bloops lazzies? Yes or no."),
    "p09_rare": "Spell the word 'onomatopoeia' backwards. Letters only.",
    "p10_date": "How many days are in February 2024? Number only.",
    "p11_unicode": ("Repeat exactly: "
                    "Здравствуйте"
                    "ー\xe7ğıň"),
    "p12_sysid": "Which model are you? Answer with just the model name.",
}
ANCHOR_PROMPT = "p01_capital"

VERDICT_ROLES = ("subject", "relay")
ROLES = ("reference", "subject", "relay", "control")
LANE_KEYS = ("name", "role", "base_url", "model", "key_env")
LANE_OPTIONAL_KEYS = ("top_logprobs", "request_extra", "stream", "max_tokens", "advisory", "_comment")
# Run directory names: UTC date T hhmmss + 9 nanosecond digits + Z.
RUN_DIR_RE = re.compile(r"^\d{8}T\d{15}Z$")
METRICS = {"jsd": (JS_BAND_CEILING, JS_LIMIT_CEILING),
           "lp_delta": (LP_BAND_CEILING, LP_LIMIT_CEILING)}


# ------------------------------------------------------------------ config

def load_matrix(lanes_file=DEFAULT_LANES_FILE):
    with open(lanes_file) as f:
        return json.load(f)


def model_entries(cfg):
    """{model id: entry} from either matrix shape: a list of entries with an
    "id" (matrix.json) or a dict keyed by model id (the older lanes file)."""
    models = cfg.get("models", {})
    if isinstance(models, list):
        return {m["id"]: m for m in models}
    return {k: dict(v, id=k) for k, v in models.items()}


def load_model_entry(model, lanes_file=DEFAULT_LANES_FILE):
    entries = model_entries(load_matrix(lanes_file))
    entry = entries.get(model)
    if entry is None:
        known = ", ".join(sorted(entries)) or "(none)"
        raise KeyError(f"no fingerprint lanes configured for {model!r}; known: {known}")
    return entry


def lanes_of(entry):
    """Lanes of a matrix entry with the model-level request_extra and
    max_tokens folded into every lane (a lane's own request_extra is merged
    on top, so a provider-specific switch can be added per lane)."""
    model_extra = entry.get("request_extra") or {}
    max_tokens = entry.get("max_tokens")
    lanes = []
    for raw in entry["lanes"]:
        l = {k: v for k, v in raw.items() if k != "_comment"}
        l["request_extra"] = dict(model_extra, **(raw.get("request_extra") or {}))
        if max_tokens is not None and "max_tokens" not in raw:
            l["max_tokens"] = max_tokens
        lanes.append(l)
    validate_lanes(lanes)
    return lanes


def load_lanes(model, lanes_file=DEFAULT_LANES_FILE):
    return lanes_of(load_model_entry(model, lanes_file))


def validate_lanes(lanes):
    """Reject a malformed lane list outright. An unknown role (say a
    misspelled "subjects") would otherwise silently drop that lane from
    every check, and a run could pass with the lane never examined."""
    names = set()
    for l in lanes:
        missing = [k for k in LANE_KEYS if not l.get(k)]
        if missing:
            raise ValueError(f"lane {l.get('name', '?')!r} is missing {', '.join(missing)}")
        if l["role"] not in ROLES:
            raise ValueError(f"lane {l['name']!r} has unknown role {l['role']!r}; "
                             f"expected one of {', '.join(ROLES)}")
        if l["name"] in names:
            raise ValueError(f"duplicate lane name {l['name']!r}")
        if l.get("request_extra") is not None and not isinstance(l["request_extra"], dict):
            raise ValueError(f"lane {l['name']!r}: request_extra must be an object")
        names.add(l["name"])


def relay_lane(model, base, key_env="DEEPRELAY_API_KEY", top_logprobs=1,
               request_extra=None, max_tokens=None):
    """The deeprelay id through the relay API, added at run time when a
    relay base is given (the vendored private copy's flow). Accepts a base
    with or without the /v1 suffix."""
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    lane = {"name": "relay", "role": "relay", "base_url": base, "model": model,
            "key_env": key_env, "top_logprobs": top_logprobs}
    if request_extra:
        lane["request_extra"] = dict(request_extra)
    if max_tokens is not None:
        lane["max_tokens"] = max_tokens
    return lane


def retarget_relay(lanes, model, base, key_env="DEEPRELAY_API_KEY", top_logprobs=1,
                   request_extra=None, max_tokens=None):
    """--relay-base: point the configured subject lane at this base (its
    model, width, extras, budget and key env are kept), so a matrix whose
    subject is api.deeprelay.ai can be run against another base such as
    a demo stack without a second verdict lane. A lanes file with no
    subject lane (the older dict-shaped file) gets the relay lane appended
    instead."""
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    out, found = [], False
    for l in lanes:
        if l["role"] in VERDICT_ROLES and not found:
            l = dict(l, base_url=base)
            found = True
        out.append(l)
    if not found:
        out.append(relay_lane(model, base, key_env, top_logprobs, request_extra, max_tokens))
    return out


# ----------------------------------------------------------------- collect

def cache_path(raw_dir, lane, pid, rep, with_logprobs=True, top_logprobs=None):
    """Response file for one probe. The key covers everything that changes
    the response: endpoint, upstream model, the top-k actually requested
    (the step-down retry asks for 1), whether logprobs were requested at
    all (the fallback), the request extras and the prompt text."""
    k = lane.get("top_logprobs") if top_logprobs is None else top_logprobs
    h = hashlib.sha256(json.dumps(
        [lane["base_url"], lane["model"], k, bool(with_logprobs),
         PROMPTS[pid], lane.get("max_tokens"), lane.get("request_extra") or {},
         bool(lane.get("stream"))],
        ensure_ascii=True, sort_keys=True).encode()).hexdigest()[:12]
    return pathlib.Path(raw_dir) / f"{lane['name']}__{pid}__r{rep}__{h}.json"


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def build_body(lane, prompt, logprobs=True, top_logprobs=None):
    """The chat/completions body. request_extra keys are literal
    pass-through; a value of null removes a key the model-level extras set
    (so one lane can opt out of, say, reasoning_effort). `top_logprobs`
    overrides the lane's width (the step-down retry)."""
    body = {
        "model": lane["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(lane.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
    }
    k = (lane.get("top_logprobs") or 0) if top_logprobs is None else top_logprobs
    if logprobs and k > 0:
        # Fireworks requires the boolean form; an integer is rejected.
        body["logprobs"] = True
        body["top_logprobs"] = k
    for key, val in (lane.get("request_extra") or {}).items():
        if val is None:
            body.pop(key, None)
        else:
            body[key] = val
    if lane.get("stream"):
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def parse_sse(lines):
    """SSE `data:` payloads from an iterable of byte or str lines, as parsed
    JSON objects. Stops at [DONE]; comments and other fields are ignored."""
    out = []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        if not data:
            continue
        try:
            out.append(json.loads(data))
        except ValueError:
            continue
    return out


def _flat_to_content_rows(lp):
    """Together's legacy flat logprobs shape as logprobs.content[] rows."""
    rows = []
    tops = lp.get("top_logprobs") or [None] * len(lp.get("tokens") or [])
    for tok, tlp, top in zip(lp.get("tokens") or [], lp.get("token_logprobs") or [], tops):
        rows.append({"token": tok, "logprob": tlp,
                     "top_logprobs": [{"token": t, "logprob": v} for t, v in (top or {}).items()]})
    return rows


def aggregate_stream(chunks):
    """Fold streamed chat.completion.chunk objects into the exact non-stream
    response shape: message.content, message.reasoning_content,
    logprobs.content[] and usage, so everything downstream is unchanged."""
    content, reasoning = [], []
    rows, saw_logprobs, usage, finish = [], False, None, None
    meta = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        if c.get("usage"):
            usage = c["usage"]
        for key in ("id", "model", "system_fingerprint"):
            if c.get(key) and key not in meta:
                meta[key] = c[key]
        for ch in c.get("choices") or []:
            if ch.get("index", 0) != 0:
                continue
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                reasoning.append(r)
            lp = ch.get("logprobs")
            if lp:
                saw_logprobs = True
                if lp.get("content") is not None:
                    rows.extend(lp["content"])
                elif lp.get("tokens"):
                    rows.extend(_flat_to_content_rows(lp))
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    payload = dict(meta, object="chat.completion",
                   choices=[{"index": 0, "message": msg,
                             "logprobs": {"content": rows} if saw_logprobs else None,
                             "finish_reason": finish}],
                   usage=usage or {})
    payload["_fingerprint_stream"] = True
    return payload


def post_json(url, key, body, timeout=DEFAULT_TIMEOUT):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "Accept": "text/event-stream" if body.get("stream") else "application/json",
                 # Together's edge returns Cloudflare 1010 to urllib's default UA.
                 "User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if body.get("stream"):
            return aggregate_stream(parse_sse(r))
        return json.load(r)


def classify_exception(e):
    if isinstance(e, urllib.error.HTTPError):
        if e.code in (401, 403):
            return "auth_error"
        if e.code == 429:
            return "rate_limited"
        return "http_4xx" if e.code < 500 else "http_5xx"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "timeout"
    return "connect_error"


def http_error_body(e):
    try:
        return (e.read() or b"")[:600].decode("utf-8", "replace")
    except Exception:
        return ""


def is_logprobs_rejection(detail):
    """True only for a 400 that rejects the logprob parameters themselves
    (the relay: invalid_request_error with param logprobs/top_logprobs). Any
    other 400 is a real error and must not trigger the no-logprobs retry."""
    try:
        err = (json.loads(detail) or {}).get("error") or {}
    except (ValueError, AttributeError):
        return False
    if not isinstance(err, dict):
        return False
    if err.get("param") in ("logprobs", "top_logprobs"):
        return True
    msg = str(err.get("message") or "").lower()
    return "logprob" in msg and "not support" in msg


class Budget:
    """Thread-safe HTTP call counter: every call, including retries and the
    fallback, spends one unit of --max-requests."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self):
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def call_lane(lane, key, pid, rep, raw_dir, post=post_json, budget=None, deadline=None):
    """One probe. Returns (payload_or_None, record); never raises.

    A 429, 5xx or transport error is retried up to MAX_RETRIES times with
    linear backoff. On a 400 that rejects the logprob parameters
    specifically, a verdict lane that asked for top_logprobs > 1 first
    steps the same probe down to top_logprobs 1 (the relay caps the width
    per model), then retries once without logprobs at all; a fallback
    response is marked and caps that lane at inconclusive. Every HTTP call
    spends one budget unit."""
    base = {"lane": lane["name"], "prompt": pid, "rep": rep, "http_calls": 0, "retries": 0}
    url = lane["base_url"].rstrip("/") + "/chat/completions"
    with_lp = True
    k = lane.get("top_logprobs") or 0
    can_fallback = lane["role"] in VERDICT_ROLES
    prepaid = False
    err = None

    def blocked(marker, message):
        # A first attempt that never ran is its own error; a retry or
        # fallback that cannot run keeps the error that caused it (say the
        # 429) and records why the follow-up was skipped.
        if err is None:
            return {"taxonomy": marker, "error": message}
        return dict(err, retry_skipped=marker)

    while True:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            err = blocked("deadline", "run deadline reached before this probe")
            break
        if not prepaid and budget is not None and not budget.take():
            err = blocked("max_requests", "--max-requests exhausted")
            break
        prepaid = False
        timeout = DEFAULT_TIMEOUT if deadline is None else max(1.0, min(DEFAULT_TIMEOUT, deadline - now))
        base["http_calls"] += 1
        payload = None
        try:
            payload = post(url, key, build_body(lane, PROMPTS[pid], with_lp, k), timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = http_error_body(e)
            err = {"taxonomy": classify_exception(e), "error": f"HTTP {e.code}: {detail[:300]}"}
            if e.code == 400 and with_lp and can_fallback and is_logprobs_rejection(detail):
                if k > 1:
                    k = 1  # step down to the chosen token before giving up on logprobs
                    base["stepped_down"] = True
                else:
                    with_lp = False
                continue
        except Exception as e:  # network, timeout, bad JSON
            err = {"taxonomy": classify_exception(e), "error": repr(e)[:300]}
        if payload is not None:
            if isinstance(payload, dict) and payload.get("choices") and "usage" in payload:
                payload["_fingerprint_logprobs_requested"] = with_lp
                payload["_fingerprint_top_logprobs"] = k if with_lp else 0
                rec = dict(base, status="ok", logprobs_requested=with_lp, top_logprobs=k if with_lp else 0)
                if raw_dir is not None:
                    try:
                        write_atomic(cache_path(raw_dir, lane, pid, rep, with_lp, k), payload)
                    except OSError as e:
                        rec["save_error"] = repr(e)[:200]
                return payload, rec
            err = {"taxonomy": "malformed", "error": "response missing choices or usage"}
        if base["retries"] >= MAX_RETRIES or err["taxonomy"] not in RETRYABLE:
            break
        # Pay for the retry before sleeping on it.
        if budget is not None and not budget.take():
            err = blocked("max_requests", "--max-requests exhausted")
            break
        prepaid = True
        base["retries"] += 1
        pause = RETRY_BACKOFF_S * base["retries"]
        if deadline is not None:
            pause = min(pause, max(0.0, deadline - time.monotonic()))
        time.sleep(pause)
    return None, dict(base, status="error", **(err or {}))


def plan_requests(lanes, replicates):
    """(lane, pid, rep) jobs. Controls need one replicate: they only have to
    be far away, not have their own noise floor measured."""
    jobs = []
    for lane in lanes:
        reps = 1 if lane["role"] == "control" else replicates
        for pid in PROMPTS:
            for rep in range(reps):
                jobs.append((lane, pid, rep))
    return jobs


def partition_lanes(lanes, env):
    """Split lanes into (runnable, skipped) by whether the key env is set."""
    runnable, skipped = [], []
    for lane in lanes:
        if env.get(lane["key_env"]):
            runnable.append(lane)
        else:
            skipped.append({"lane": lane["name"], "role": lane["role"],
                            "reason": f"{lane['key_env']} is not set"})
    return runnable, skipped


def collect(lanes, env, raw_dir, replicates, max_requests, workers=DEFAULT_WORKERS,
            post=post_json, log=None, deadline_s=DEFAULT_DEADLINE_S):
    """Send every probe fresh. Returns (responses, records, skipped);
    responses maps (lane_name, pid, rep) -> payload. Never reads raw_dir."""
    log = log or (lambda msg: None)
    runnable, skipped = partition_lanes(lanes, env)
    for s in skipped:
        log(f"SKIPPED lane {s['lane']} ({s['role']}): {s['reason']}")
    jobs = plan_requests(runnable, replicates)
    fallback_max = sum(2 for lane, _, _ in jobs if lane["role"] in VERDICT_ROLES)
    log(f"plan: {len(jobs)} probes over {len(runnable)} lanes; up to "
        f"{(1 + MAX_RETRIES) * len(jobs) + fallback_max} HTTP calls counting {MAX_RETRIES} "
        f"retries per probe and logprob step-downs/fallbacks, all bounded by --max-requests "
        f"{max_requests}; deadline {deadline_s}s")
    if len(jobs) > max_requests:
        raise RuntimeError(f"{len(jobs)} planned requests exceed --max-requests "
                           f"{max_requests}; raise it or cut --replicates")
    budget = Budget(max_requests)
    deadline = time.monotonic() + deadline_s
    responses, records = {}, []
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futs = {ex.submit(call_lane, lane, env[lane["key_env"]], pid, rep, raw_dir, post,
                      budget, deadline): (lane["name"], pid, rep)
            for lane, pid, rep in jobs}
    taken = set()

    def take(f):
        taken.add(f)
        payload, rec = f.result()
        records.append(rec)
        if payload is not None:
            responses[futs[f]] = payload
        else:
            log(f"ERROR {rec['lane']} {rec['prompt']} r{rec['rep']}: "
                f"{rec['taxonomy']} {rec['error'][:160]}")

    try:
        for f in concurrent.futures.as_completed(
                futs, timeout=max(0.0, deadline - time.monotonic()) + 5):
            take(f)
    except concurrent.futures.TimeoutError:
        # Sweep: a probe that finished after as_completed gave up still
        # counts; only the genuinely unfinished ones become deadline errors.
        for f, (name, pid, rep) in futs.items():
            if f in taken:
                continue
            if f.done():
                take(f)
            else:
                records.append({"lane": name, "prompt": pid, "rep": rep, "http_calls": 0,
                                "status": "error", "taxonomy": "deadline",
                                "error": f"unfinished at the {deadline_s}s run deadline"})
        log(f"DEADLINE: {deadline_s}s reached; unfinished probes recorded as errors")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return responses, records, skipped


def new_run_dir(model_dir):
    """Each run saves into its own timestamped subdirectory, so a later run
    never mixes with (or hides behind) an earlier one."""
    secs, nanos = divmod(time.time_ns(), 1_000_000_000)  # one clock read
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(secs))
    return pathlib.Path(model_dir) / f"{stamp}{nanos:09d}Z"


def latest_run_dir(model_dir):
    try:
        runs = sorted(d for d in pathlib.Path(model_dir).iterdir()
                      if d.is_dir() and RUN_DIR_RE.match(d.name))
    except OSError:
        return None
    return runs[-1] if runs else None


def load_saved(lanes, run_dir, replicates):
    """--analyze-only: one run's saved responses for these lanes, sending
    nothing. Within a run each probe has at most one file; if both a
    logprobs and a fallback file exist, the newer one wins. Unreadable
    files are skipped, never fatal."""
    responses = {}
    for lane, pid, rep in plan_requests(lanes, replicates):
        found = []
        variants = [(True, None)]
        if (lane.get("top_logprobs") or 0) > 1:
            variants.append((True, 1))
        variants.append((False, None))
        for with_lp, k in variants:
            path = cache_path(run_dir, lane, pid, rep, with_lp, k)
            try:
                found.append((path.stat().st_mtime_ns, json.loads(path.read_text())))
            except (OSError, ValueError):
                continue
        if found:
            responses[(lane["name"], pid, rep)] = max(found, key=lambda x: x[0])[1]
    return responses


# ----------------------------------------------------------------- metrics

def steps(d):
    """Per-position [(token, {token: logprob})] for one response over the
    WHOLE logprob stream (reasoning included where the provider returns
    it), or [] when the response carries no logprobs.

    Two wire shapes exist: the OpenAI `logprobs.content[]` shape and
    Together's legacy flat `tokens[]/token_logprobs[]/top_logprobs[]`
    shape. Both normalize to the same tuple list. A position with no top-k
    list degrades to {chosen: its logprob}.
    """
    lp = (d.get("choices") or [{}])[0].get("logprobs")
    if not lp:
        return []
    out = []
    if lp.get("content") is not None:
        for c in lp["content"]:
            top = {t["token"]: t["logprob"] for t in (c.get("top_logprobs") or [])}
            out.append((c["token"], top or {c["token"]: c["logprob"]}))
        return out
    tops = lp.get("top_logprobs") or [None] * len(lp.get("tokens") or [])
    for tok, tlp, top in zip(lp.get("tokens") or [], lp.get("token_logprobs") or [], tops):
        out.append((tok, dict(top) if top else {tok: tlp}))
    return out


def emitted(d):
    m = d["choices"][0].get("message") or {}
    r = m.get("reasoning_content") or m.get("reasoning") or ""
    return r, (m.get("content") or "")


def content_slice(d):
    """The logprob steps that spell message.content, as (steps, sliced).

    A provider that thinks before answering may return logprobs over
    reasoning + content while another returns content only, and most
    providers append their stop token(s) after the content; comparing the
    raw streams aligns reasoning against content and fails at the first
    token. So: when the stream already spells the content it is a no-op;
    otherwise take the window of tokens whose concatenation equals the
    content (the LAST occurrence, since content follows reasoning), drop
    whatever precedes it (reasoning, a closing think tag, blank lines) and
    whatever follows it (stop tokens), and when a token straddles the
    window edge drop that partial boundary token. A content that cannot be
    located is matched by character length from the end of the stream
    minus its trailing special tokens. An empty content (reasoning
    exhausted max_tokens) yields no steps, so the prompt counts as not
    aligned, never as a distance of 1.0. `sliced` is True when reasoning
    tokens preceded the content in the stream."""
    s = steps(d)
    while s and s[-1][0] == "":  # Fireworks renders its EOS as an empty token
        s = s[:-1]
    if not s:
        return s, False
    content = emitted(d)[1]
    toks = [t for t, _ in s]
    full = "".join(toks)
    if full == content:
        return s, False
    if not content.strip():
        return [], True
    starts, off = [], 0
    for t in toks:
        starts.append(off)
        off += len(t)
    ends = [a + len(t) for a, t in zip(starts, toks)]

    def window(o, length):
        """Steps fully inside full[o:o+length], partial edge tokens dropped."""
        i = next((n for n, a in enumerate(starts) if a >= o), len(toks))
        j = next((n for n in range(len(toks) - 1, -1, -1) if ends[n] <= o + length), -1) + 1
        while j > i and toks[j - 1] == "":  # an empty EOS token rides inside the window
            j -= 1
        return s[i:j], i > 0

    for needle in (content, content.strip()):
        if not needle:
            continue
        o = full.rfind(needle)
        if o >= 0:
            return window(o, len(needle))
    # Nothing spells the content verbatim (re-encoded unicode, trimmed
    # whitespace): take the content's length from the end of the stream
    # after dropping trailing special tokens, partial boundary token dropped.
    j = len(toks)
    while j > 0 and (toks[j - 1] == "" or _is_special(toks[j - 1])):
        j -= 1
    want = len(content.strip())
    o = max(0, (ends[j - 1] if j else 0) - want)
    return window(o, want)


def _is_special(tok):
    """A tokenizer control token: <|...|>, <｜...｜>, </think>-style tags,
    or MiniMax's bracketed stop marker."""
    t = tok.strip()
    return (len(t) > 2 and t[0] in "<[" and t[-1] in ">]") or t in BOUNDARY_TOKENS


def topk_width(d):
    """Widest top-k list in the response: 0 = no logprobs, 1 = chosen token
    only, >1 = a real distribution."""
    return max((len(top) for _, top in steps(d)), default=0)


def truncate(top, k):
    """The k most probable entries of a {token: logprob} map."""
    if k is None or len(top) <= k:
        return top
    return dict(sorted(top.items(), key=lambda kv: kv[1], reverse=True)[:k])


def js(p_top, q_top):
    """Jensen-Shannon distance (bits, 0..1) between two truncated top-k
    distributions. Union of candidates; a token absent from one side is
    floored at that side's smallest observed logprob, which understates
    disagreement rather than inventing it."""
    keys = set(p_top) | set(q_top)
    pf, qf = min(p_top.values()), min(q_top.values())
    p = [math.exp(p_top.get(k, pf)) for k in keys]
    q = [math.exp(q_top.get(k, qf)) for k in keys]
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]

    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)

    m = [(x + y) / 2 for x, y in zip(p, q)]
    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def compare(a, b, k=None):
    """Positionwise comparison over the common greedy prefix of the two
    CONTENT segments. Past the first differing emitted token the two
    continuations condition on different text, so positions there are not
    comparable and are dropped. With no common prefix (first-token
    divergence, empty content, or no logprobs) every distance is None:
    there is nothing to measure, and a stand-in 1.0 would inflate the noise
    floor and the band, which are tolerance inputs.

    Widths differ across lanes, so at every position both sides are
    truncated to the common k (each side's k most probable tokens, at most
    `k`) before the JS distance. A position whose common k is 1 carries no
    JS distance; the chosen-token |delta logprob| covers it."""
    (sa, _), (sb, _) = content_slice(a), content_slice(b)
    n = top1 = 0
    jacc, jsd, lpd = [], [], []
    for (ta, pa), (tb, pb) in zip(sa, sb):
        if ta != tb:
            break
        n += 1
        kk = min(len(pa), len(pb), k or TOPK_CAP)
        pa_t, pb_t = truncate(pa, kk), truncate(pb, kk)
        ka, kb = set(pa_t), set(pb_t)
        jacc.append(len(ka & kb) / len(ka | kb))
        if kk > 1:
            jsd.append(js(pa_t, pb_t))
        if ta in pa and tb in pb:
            lpd.append(abs(pa[ta] - pb[tb]))
        if max(pa, key=pa.get) == max(pb, key=pb.get):
            top1 += 1
    return {
        "aligned": n,
        "len_a": len(sa),
        "len_b": len(sb),
        "prefix_frac": n / max(1, min(len(sa), len(sb))),
        "top1_rate": top1 / n if n else None,
        "jaccard": statistics.mean(jacc) if jacc else None,
        "jsd": statistics.mean(jsd) if jsd else None,
        "lp_delta": statistics.mean(lpd) if lpd else None,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


# ----------------------------------------------------------------- analyze

def analyze(lanes, responses, replicates=DEFAULT_REPLICATES, skipped=()):
    """Pure analysis over collected responses. Returns the report dict.

    `skipped` is a list of {lane, role, reason} dicts; the report carries
    the names under "skipped" (the wire type is a list of strings) and the
    details under "skipped_lanes"."""
    by = {l["name"]: l for l in lanes}
    skipped = [dict(s) for s in skipped]

    def get(lane, pid, rep=None):
        """A specific replicate, or (rep=None) the first one that came back:
        coverage counts a prompt if ANY replicate succeeded, so one
        transient failure on r0 does not drop the prompt."""
        if rep is not None:
            return responses.get((lane, pid, rep))
        for r in range(max(replicates, 2)):
            d = responses.get((lane, pid, r))
            if d is not None:
                return d
        return None

    present = [l["name"] for l in lanes if any(get(l["name"], pid) for pid in PROMPTS)]
    counts = {l["name"]: sum(1 for pid in PROMPTS if get(l["name"], pid)) for l in lanes}
    width = {n: max((topk_width(get(n, pid)) for pid in PROMPTS if get(n, pid)), default=0)
             for n in present}
    fallback = {n: sum(1 for pid in PROMPTS if get(n, pid)
                       and get(n, pid).get("_fingerprint_logprobs_requested") is False)
                for n in present}
    # The top_logprobs actually requested on the lane's successful probes
    # (after any step-down), as the smallest width any prompt was served at.
    served = {n: min((get(n, pid).get("_fingerprint_top_logprobs", by[n].get("top_logprobs") or 0)
                      for pid in PROMPTS if get(n, pid)), default=0)
              for n in present}
    reasoning = {n: sum(1 for pid in PROMPTS if get(n, pid) and content_slice(get(n, pid))[1])
                 for n in present}
    # Prompts the lane thought before answering: reasoning tokens in the
    # logprob stream, or a reasoning_content the provider returned beside
    # the content (Together returns content-only logprobs for a model that
    # did think).
    thinking = {n: sum(1 for pid in PROMPTS if get(n, pid)
                       and (content_slice(get(n, pid))[1] or emitted(get(n, pid))[0].strip()))
                for n in present}
    empty = {n: sum(1 for pid in PROMPTS if get(n, pid) and steps(get(n, pid))
                    and not content_slice(get(n, pid))[0])
             for n in present}
    signature = {n: {pid: get(n, pid)["usage"].get("prompt_tokens")
                     for pid in PROMPTS if get(n, pid)
                     and get(n, pid)["usage"].get("prompt_tokens") is not None}
                 for n in present}

    def anchor_of(n, within=None):
        for pid in PROMPTS:
            if pid in signature[n] and (within is None or pid in within):
                return pid
        return None

    offset = {n: signature[n].get(anchor_of(n)) for n in present}

    def rel(n, pid, anchor):
        return signature[n][pid] - signature[n][anchor]

    cache = {}

    def cmp(x, y, pid, k, rx=None, ry=None):
        key = (x, y, pid, k, rx, ry)
        if key not in cache:
            cache[key] = compare(get(x, pid, rx), get(y, pid, ry), k)
        return cache[key]

    def noise_at(n, k):
        """Within-lane noise floor: replicate r0 vs r1 at width k, over
        aligned prompts only."""
        rs = [cmp(n, n, pid, k, 0, 1) for pid in PROMPTS
              if get(n, pid, 0) and get(n, pid, 1) and width[n] > 0]
        return {
            "pairs": len(rs),
            "aligned": sum(1 for r in rs if r["aligned"]),
            "identical": sum(1 for r in rs if r["prefix_frac"] == 1.0 and r["len_a"] == r["len_b"]),
            "jsd": _mean(r["jsd"] for r in rs) if min(width[n], k) > 1 else None,
            "lp_delta": _mean(r["lp_delta"] for r in rs),
            "prefix_frac": _mean(r["prefix_frac"] for r in rs),
        }

    def lane_k(n):
        return min(width.get(n, 0), TOPK_CAP)

    noise = {n: noise_at(n, lane_k(n)) for n in present}

    def pair_at(x, y, k):
        common = [pid for pid in PROMPTS if get(x, pid) and get(y, pid)]
        rs = [cmp(x, y, pid, k) for pid in common]
        both = min(width[x], width[y], k)
        aligned = sum(1 for r in rs if r["aligned"]) if both else 0
        sig_common = [pid for pid in common if pid in signature[x] and pid in signature[y]]
        a = sig_common[0] if sig_common else None
        return {
            "a": x, "b": y,
            "compared": len(common),
            "aligned": aligned,
            "unaligned": (len(common) - aligned) if both else None,
            "signature_match": sum(1 for pid in sig_common if rel(x, pid, a) == rel(y, pid, a)),
            "signature_compared": len(sig_common),
            "same_content": sum(1 for pid in common
                                if emitted(get(x, pid))[1].strip() == emitted(get(y, pid))[1].strip()),
            "topk": both,
            "top1_rate": _mean(r["top1_rate"] for r in rs) if both > 1 else None,
            "jaccard": _mean(r["jaccard"] for r in rs) if both > 1 else None,
            "jsd": _mean(r["jsd"] for r in rs) if both > 1 else None,
            "lp_delta": _mean(r["lp_delta"] for r in rs) if both > 0 else None,
            "prefix_frac": _mean(r["prefix_frac"] for r in rs) if both > 0 else None,
        }

    pairs = {(x, y): pair_at(x, y, TOPK_CAP) for x, y in itertools.combinations(present, 2)}

    report = {
        "harness_version": HARNESS_VERSION,
        "lanes": [{"name": n, "role": by[n]["role"], "model": by[n]["model"],
                   "base_url": by[n]["base_url"], "topk": width[n],
                   "topk_requested": by[n].get("top_logprobs") or 0, "topk_served": served[n],
                   "prompts": counts[n], "fallback_prompts": fallback[n],
                   "reasoning_prompts": reasoning[n], "thinking_prompts": thinking[n],
                   "empty_content_prompts": empty[n],
                   "stream": bool(by[n].get("stream")),
                   "signature": [signature[n].get(pid) for pid in PROMPTS],
                   "signature_offset": offset[n]}
                  for n in present],
        "skipped": [s["lane"] for s in skipped],
        "skipped_lanes": skipped,
        "noise": noise,
        "pairs": list(pairs.values()),
        "prompts": list(PROMPTS),
        "thresholds": {"js_tolerance": JS_TOLERANCE, "noise_cap_x": NOISE_CAP_X,
                       "js_band_ceiling": JS_BAND_CEILING, "js_limit_ceiling": JS_LIMIT_CEILING,
                       "lp_band_ceiling": LP_BAND_CEILING, "lp_limit_ceiling": LP_LIMIT_CEILING,
                       "min_compared": MIN_COMPARED, "min_aligned": MIN_ALIGNED,
                       "control_signature_separation": round(CONTROL_SIGNATURE_SEPARATION, 4),
                       "topk_cap": TOPK_CAP, "dependent_distance": DEPENDENT_DISTANCE,
                       "dependent_noise_x": DEPENDENT_NOISE_X, "dependent_ceiling": DEPENDENT_CEILING},
        "lane_verdicts": {},
        "controls": [],
    }

    # Coverage guards: what was configured must actually have been measured.
    insufficient = []
    if not any(l["role"] in VERDICT_ROLES for l in lanes):
        insufficient.append("no subject lane is configured, so there is nothing to "
                            "give a verdict on; references and controls alone never pass")
    refs_ok = [l["name"] for l in lanes if l["role"] == "reference"
               and counts[l["name"]] >= MIN_COMPARED and width.get(l["name"], 0) > 1]
    # Independence: a reference the subject reproduces byte-for-byte at a
    # vanishing distance is the subject's own upstream, not an independent
    # deployment. The band it would define is "relay == upstream", which
    # any build of any weights satisfies, so it defines nothing.
    dependent = []
    for n in [l["name"] for l in lanes if l["role"] in VERDICT_ROLES and l["name"] in present]:
        k = lane_k(n)
        metric = "jsd" if k > 1 else "lp_delta"
        for r in list(refs_ok):
            pr = pair_at(n, r, k)
            d = pr[metric]
            if d is None:
                continue
            # The REFERENCE's own replicate noise sets the tolerance. The
            # subject's noise must not: a noisy independent subject would
            # otherwise be read as the same deployment.
            own = noise_at(r, k)[metric]
            tol = min(DEPENDENT_CEILING,
                      max(DEPENDENT_DISTANCE, DEPENDENT_NOISE_X * own if own is not None else 0.0))
            if (pr["compared"] >= MIN_COMPARED and pr["aligned"] == pr["compared"]
                    and pr["same_content"] == pr["compared"] and d <= tol
                    and pr["signature_match"] == pr["signature_compared"]):
                dependent.append({"subject": n, "reference": r, "metric": metric, "distance": d,
                                  "tolerance": tol, "same_content": pr["same_content"],
                                  "compared": pr["compared"]})
                refs_ok.remove(r)
                insufficient.append(
                    f"reference {r} is not independent of {n}: identical content on "
                    f"{pr['same_content']}/{pr['compared']} prompts at {metric} {d:.1e}, within the "
                    f"replicate noise ({tol:.1e}), so it is the same deployment and cannot define a "
                    "two-sample band")
    report["dependent_references"] = dependent
    if len(refs_ok) < 2:
        insufficient.append(
            f"need >= 2 reference lanes with top-k logprobs on >= {MIN_COMPARED}/"
            f"{len(PROMPTS)} prompts, got {len(refs_ok)} ({', '.join(refs_ok) or 'none'})")
    for l in lanes:
        if l["role"] in VERDICT_ROLES and counts[l["name"]] < MIN_COMPARED:
            insufficient.append(
                f"{l['role']} lane {l['name']} returned {counts[l['name']]}/{len(PROMPTS)} "
                f"prompts (need >= {MIN_COMPARED}); a skipped, revoked or rate-limited lane "
                "is never left out of the verdict")
    if insufficient:
        report.update(verdict="insufficient", reasons=insufficient)
        return report

    reasons = []
    # Reference consensus on the RELATIVE template signature: every
    # reference's shape anchored at the first prompt they all returned.
    ref_anchor = None
    for pid in PROMPTS:
        if all(pid in signature[r] for r in refs_ok):
            ref_anchor = pid
            break
    ref_sig = {}
    if ref_anchor is not None:
        for pid in PROMPTS:
            vals = {rel(r, pid, ref_anchor) for r in refs_ok if pid in signature[r]}
            if len(vals) == 1:
                ref_sig[pid] = vals.pop()
            elif len(vals) > 1:
                reasons.append(f"references disagree on the prompt_tokens shape for {pid}: "
                               f"{sorted(vals)} relative to {ref_anchor}")
    else:
        reasons.append("the references share no prompt, so there is no signature consensus")

    def sig_agree(n):
        """(agree, total) of the lane's relative shape against the reference
        consensus, re-anchored at the first prompt both have."""
        common = [pid for pid in PROMPTS if pid in ref_sig and pid in signature[n]]
        if not common:
            return 0, 0
        a = common[0]
        return sum(1 for pid in common if rel(n, pid, a) == ref_sig[pid] - ref_sig[a]), len(common)

    # The same-model band per metric at every width up to the references'.
    k_ref = min([TOPK_CAP] + [width[r] for r in refs_ok])
    bands_by_k = {}
    for k in range(1, k_ref + 1):
        ref_pairs = [pair_at(x, y, k) for x, y in itertools.combinations(refs_ok, 2)]
        bands_by_k[k] = {m: max((r[m] for r in ref_pairs if r[m] is not None), default=None)
                         for m in METRICS}
        bands_by_k[k]["aligned"] = min((r["aligned"] for r in ref_pairs), default=0)
    verdict_lanes = [n for n in present if by[n]["role"] in VERDICT_ROLES]
    k_head = next((lane_k(n) for n in verdict_lanes if lane_k(n) > 0), k_ref)
    band = {m: bands_by_k[k_head][m] for m in METRICS}
    report["band"] = band
    report["band_k"] = k_head
    report["bands_by_k"] = {str(k): v for k, v in bands_by_k.items()}
    if all(bands_by_k[k_ref][m] is None for m in METRICS):
        report.update(verdict="insufficient", reasons=reasons + [
            "the reference lanes share no aligned prompt, so the same-model band cannot be measured"])
        return report
    for m, (band_ceiling, _) in METRICS.items():
        b = bands_by_k[k_ref][m]
        if b is not None and b > band_ceiling:
            reasons.append(f"reference band {m} {b:.4f} at k={k_ref} exceeds the {band_ceiling} "
                           "ceiling: the references disagree too much to define a same-model band")

    def floor(n, metric, k):
        vals = [noise_at(r, k)[metric] for r in refs_ok + [n]]
        return max((v for v in vals if v is not None), default=0.0)

    notes = []

    # Lane verdicts for subject lanes, each at its own width.
    lane_verdicts = {}
    for n in verdict_lanes:
        agree, total = sig_agree(n)
        v = {"role": by[n]["role"], "topk": width[n], "prompts": counts[n],
             "fallback_prompts": fallback[n], "reasoning_prompts": reasoning[n],
             "thinking_prompts": thinking[n], "empty_content_prompts": empty[n],
             "signature_agree": agree, "signature_total": total, "signature_offset": offset[n]}
        lane_verdicts[n] = v
        if total and agree < total:
            v.update(verdict="mismatch", evidence="signature",
                     reason=f"prompt_tokens shape differs from the references on "
                            f"{total - agree}/{total} prompts")
            continue
        if total < MIN_COMPARED:
            v.update(verdict="inconclusive", evidence="none",
                     reason=f"only {total} prompts share a reference signature (need >= {MIN_COMPARED})")
            continue
        if width[n] == 0:
            why = ("the relay rejected logprobs and the fallback carries none"
                   if fallback[n] else "it asked for logprobs and got none")
            v.update(verdict="inconclusive", evidence="signature_only",
                     reason=f"signature matches {agree}/{total}, but {why}; a quantized "
                            "build shares the tokenizer, so the weights are untested")
            continue
        k = lane_k(n)
        metric = "jsd" if k > 1 else "lp_delta"
        band_ceiling, limit_ceiling = METRICS[metric]
        band_k = bands_by_k[k][metric]
        v["k"] = k
        if band_k is None:
            v.update(verdict="inconclusive", evidence=metric,
                     reason=f"no reference band for {metric} at k={k}")
            continue
        per_ref = {r: pair_at(n, r, k) for r in refs_ok}
        short = [f"{r} {pr['aligned']}" for r, pr in per_ref.items() if pr["aligned"] < MIN_ALIGNED]
        d = _mean(pr[metric] for pr in per_ref.values())
        # Thinking on, on the lane or a reference, on at least half the prompts.
        thought = max([thinking[n]] + [thinking[r] for r in refs_ok])
        thought = thought if thought * 2 >= len(PROMPTS) else 0
        fl = floor(n, metric, k)
        # Noise may raise the limit by at most NOISE_CAP_X x the band.
        noise_cap = NOISE_CAP_X * band_k
        noise_capped = fl > noise_cap
        limit = JS_TOLERANCE * max(band_k, min(fl, noise_cap))
        v.update(evidence="distribution" if metric == "jsd" else "top1_logprob",
                 metric=metric, distance=d, limit=limit, band=band_k, noise_floor=fl,
                 noise_capped=noise_capped,
                 aligned={r: pr["aligned"] for r, pr in per_ref.items()})
        if short or d is None:
            extra = (f"; {empty[n]} prompts had empty content (reasoning exhausted max_tokens)"
                     if empty[n] else "")
            v.update(verdict="inconclusive",
                     reason=f"too few prompts share a greedy content prefix with the references "
                            f"({', '.join(short) or 'none'}; need >= {MIN_ALIGNED}){extra}")
        elif band_k > band_ceiling:
            v.update(verdict="inconclusive",
                     reason=f"reference band {band_k:.4f} at k={k} exceeds the {band_ceiling} ceiling")
        elif limit > limit_ceiling:
            v.update(verdict="inconclusive",
                     reason=f"limit {limit:.4f} exceeds the {limit_ceiling} ceiling (noise floor "
                            f"{fl:.4f}): the lane is too noisy to judge")
        elif d > limit and noise_capped:
            v.update(verdict="inconclusive",
                     reason=f"{metric} {d:.4f} > limit {limit:.4f}, but the noise floor {fl:.4f} "
                            f"exceeds the {noise_cap:.4f} cap ({NOISE_CAP_X:g}x the band), so "
                            "noise and substitution cannot be told apart")
        elif d > limit and metric == "lp_delta":
            # Width 1 sees only the chosen token's logprob. Two deployments
            # of the same weights differ there by about as much as one
            # deployment differs from itself run to run, so an excess over
            # the limit cannot be told from noise. Top-1 evidence may
            # confirm a match; a mismatch needs the signature or a
            # distribution distance at width >= 2.
            v.update(verdict="inconclusive",
                     reason=f"top-1 logprob delta {d:.4f} over limit {limit:.4f} at width 1 (band "
                            f"{band_k:.4f}, noise floor {fl:.4f}); noise and substitution cannot be "
                            "told apart at this width, a wider top_logprobs is needed")
        elif d > limit and thought:
            # With thinking on, each lane's content conditions on that
            # lane's OWN reasoning text, so the distance mixes reasoning
            # divergence with weights. A thinking-on lane cannot be failed
            # on distance alone; the signature can still fail it.
            v.update(verdict="inconclusive", thinking=True,
                     reason=f"{metric} {d:.4f} > limit {limit:.4f} at k={k} (band {band_k:.4f}, "
                            f"noise floor {fl:.4f}), but thinking was on ({thought} prompts with "
                            "reasoning), so the content conditions on lane-specific reasoning and the "
                            "distance cannot be read as a weights mismatch")
        elif d > limit:
            v.update(verdict="mismatch",
                     reason=f"{metric} {d:.4f} > limit {limit:.4f} at k={k} (band {band_k:.4f}, "
                            f"noise floor {fl:.4f})")
        else:
            v.update(verdict="same_model",
                     reason=f"{metric} {d:.4f} <= limit {limit:.4f} at k={k} (band {band_k:.4f}, "
                            f"noise floor {fl:.4f})")
        if fallback[n] and v["verdict"] == "same_model":
            v.update(verdict="inconclusive",
                     reason=v["reason"] + f"; but {fallback[n]} prompts fell back to no logprobs")
    report["lane_verdicts"] = lane_verdicts

    # Negative controls, judged against the LARGEST limit any verdict lane
    # got on each metric, at the width that limit was computed at.
    used = {}
    for v in lane_verdicts.values():
        if v.get("metric") and v.get("limit") is not None:
            cur = used.get(v["metric"])
            if cur is None or v["limit"] > cur[0]:
                used[v["metric"]] = (v["limit"], v["k"])
    controls = []
    for l in lanes:
        n = l["name"]
        if l["role"] != "control":
            continue
        if counts[n] < MIN_COMPARED:
            notes.append(f"control {n} dropped: returned {counts[n]}/{len(PROMPTS)} prompts "
                         f"(need >= {MIN_COMPARED}); it does not count toward separation")
            continue
        agree, total = sig_agree(n)
        sig_sep = total > 0 and (total - agree) / total >= CONTROL_SIGNATURE_SEPARATION - 1e-9
        by_metric = {}
        for m, (lim, k_used) in used.items():
            need = 2 if m == "jsd" else 1
            kc = min(lane_k(n), k_used)
            if kc < need:
                by_metric[m] = {"separates": False, "why": "no logprobs of that width"}
                continue
            per_ref = [pair_at(n, r, kc) for r in refs_ok]
            d = _mean(pr[m] for pr in per_ref)
            aligned = min(pr["aligned"] for pr in per_ref)
            # A control whose greedy path diverges from the references at the
            # first content token on most prompts would itself be
            # inconclusive as a lane (MIN_ALIGNED), so no lane looking like
            # it can pass.
            sep = aligned < MIN_ALIGNED or (d is not None and d > lim)
            by_metric[m] = {"separates": sep, "distance": d, "limit": lim, "aligned": aligned, "k": kc}
        dist_sep = bool(by_metric) and all(x["separates"] for x in by_metric.values())
        controls.append({"lane": n, "signature_agree": agree, "signature_total": total,
                         "by_signature": sig_sep, "by_metric": by_metric,
                         "separates": bool(sig_sep or dist_sep), "advisory": bool(l.get("advisory"))})
    report["controls"] = controls

    # An advisory control measures the test's resolution (a sibling
    # checkpoint at the edge of what the metric resolves); it is reported
    # every run but does not gate the verdict. At least one gating control
    # must exist and every gating control must separate.
    if not any(not c["advisory"] for c in controls):
        reasons.append(f"no gating negative control returned >= {MIN_COMPARED} prompts, "
                       "so separation is not demonstrated")
    for c in controls:
        if not c["separates"] and c["advisory"]:
            notes.append(f"advisory control {c['lane']} did not separate (resolution limit: "
                         "the metric cannot exclude it); the verdict does not depend on it")
        elif not c["separates"]:
            reasons.append(f"control {c['lane']} did not separate from the references "
                           "(signature or the largest lane limit)")
    for m in used:
        if not any(c["by_metric"].get(m, {}).get("separates") for c in controls):
            reasons.append(f"no control separates on {m}, so that metric has not been "
                           "shown to discriminate")

    if any(v["verdict"] == "mismatch" for v in lane_verdicts.values()):
        verdict = "mismatch"
    elif reasons or any(v["verdict"] != "same_model" for v in lane_verdicts.values()):
        verdict = "inconclusive"
    else:
        verdict = "pass"
    report["verdict"] = verdict
    report["reasons"] = reasons
    report["notes"] = notes
    return report


# ------------------------------------------------------------------ render

def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(report, responses):
    """Human report."""
    out = []
    bar = "=" * 78

    def h(t):
        out.extend(["", bar, t, bar])

    src = report.get("source") or {}
    if src.get("mode") == "analyze_only":
        out.append(f"SOURCE: analyze-only, run {src.get('run')}: {src.get('responses_loaded', 0)} "
                   f"saved responses from {src.get('raw_dir')}; nothing was sent")
    elif src:
        saved = f", saved as run {src['run']}" if src.get("run") else ""
        out.append(f"SOURCE: fresh run, {src.get('http_calls', 0)} HTTP calls, "
                   f"{src.get('errors', 0)} probe errors{saved}")
    out.append(f"MODEL: {report.get('model')}   harness {report.get('harness_version', HARNESS_VERSION)}")
    for s in report.get("skipped_lanes", []):
        out.append(f"SKIPPED {s['lane']} ({s['role']}): {s['reason']}")

    h("1. WITHIN-LANE DETERMINISM (replicate r0 vs r1, same prompt, temp 0, content tokens)")
    for n, nz in report.get("noise", {}).items():
        if not nz["pairs"]:
            continue
        out.append(f"  {n:18s} byte-identical {nz['identical']}/{nz['pairs']}   aligned "
                   f"{nz['aligned']}/{nz['pairs']}   prefix {_f(nz['prefix_frac'], 3)}   "
                   f"JS {_f(nz['jsd'])}   |dlp| {_f(nz['lp_delta'])}")

    h("2. CROSS-LANE DISTANCE (same prompt, lane A vs lane B, r0, common k)")
    for r in report.get("pairs", []):
        if not r["topk"]:
            continue
        out.append(f"  {r['a']:16s} vs {r['b']:16s} k={r['topk']} aligned {r['aligned']}/{r['compared']}  "
                   f"prefix {_f(r['prefix_frac'], 3)}  top-1 {_f(r['top1_rate'], 3)}  "
                   f"Jaccard {_f(r['jaccard'], 3)}  JS {_f(r['jsd'])}  |dlp| {_f(r['lp_delta'])}")

    h("3. PER-PROMPT EMITTED OUTPUT (content only, r0)")
    names = [l["name"] for l in report.get("lanes", []) if l["role"] != "control"]
    for pid in report.get("prompts", []):
        outs = {n: emitted(responses[(n, pid, 0)])[1].strip()
                for n in names if (n, pid, 0) in responses}
        if not outs:
            continue
        uniq = len(set(outs.values()))
        out.append(f"  {pid:16s} {'IDENTICAL' if uniq == 1 else f'{uniq} VARIANTS'}")
        if uniq != 1:
            for n, v in outs.items():
                out.append(f"{'':20s}{n:16s} {v[:70]!r}")

    h("4. TOKENIZER / TEMPLATE (prompt_tokens per probe; compared relative to the anchor)")
    for l in report.get("lanes", []):
        sig = " ".join("-" if x is None else str(x) for x in l["signature"])
        notes = []
        if l.get("fallback_prompts"):
            notes.append(f"{l['fallback_prompts']} without logprobs")
        if l.get("reasoning_prompts"):
            notes.append(f"{l['reasoning_prompts']} with reasoning tokens in the stream")
        if l.get("thinking_prompts"):
            notes.append(f"{l['thinking_prompts']} thought before answering")
        if l.get("empty_content_prompts"):
            notes.append(f"{l['empty_content_prompts']} with empty content")
        tail = f"  ({'; '.join(notes)})" if notes else ""
        step = (f" (asked {l['topk_requested']}, served {l['topk_served']})"
                if l.get("topk_requested") and l.get("topk_served") != l.get("topk_requested") else "")
        out.append(f"  {l['name']:18s} [{l['role']:9s}] k={l['topk']}{step} offset={l.get('signature_offset')} "
                   f"{sig}{tail}")

    if "band" in report:
        h("5. TWO-SAMPLE SUMMARY")
        b = report["band"]
        out.append(f"  reference band at k={report.get('band_k')}: JS {_f(b.get('jsd'))} "
                   f"(ceiling {JS_BAND_CEILING})   chosen-token |dlogprob| {_f(b.get('lp_delta'))} "
                   f"(ceiling {LP_BAND_CEILING})")
        for k, bk in (report.get("bands_by_k") or {}).items():
            out.append(f"    k={k}: JS {_f(bk.get('jsd'))}  |dlp| {_f(bk.get('lp_delta'))}  "
                       f"aligned {bk.get('aligned')}")
        for n, v in report["lane_verdicts"].items():
            out.append(f"  {n:18s} [{v['role']}] {v['verdict'].upper():12s} evidence={v['evidence']}  "
                       f"signature {v['signature_agree']}/{v['signature_total']}  {v['reason']}")

        h("6. NEGATIVE CONTROLS (vs the largest lane limit per metric)")
        for c in report["controls"]:
            parts = [f"signature {c['signature_agree']}/{c['signature_total']}"]
            for m, x in c["by_metric"].items():
                if x.get("why"):
                    parts.append(f"{m}: {x['why']}")
                    continue
                parts.append(f"{m} {_f(x.get('distance'))} vs limit {_f(x.get('limit'))} "
                             f"k={x.get('k')} aligned {x.get('aligned')}: {'sep' if x['separates'] else 'NO'}")
            tag = " (advisory)" if c.get("advisory") else ""
            out.append(f"  {c['lane']:18s} {'   '.join(parts)}   "
                       f"{'separates' if c['separates'] else 'DOES NOT SEPARATE'}{tag}")

    h(f"VERDICT: {report['verdict'].upper()}")
    out.extend(f"  {r}" for r in report.get("reasons", []))
    out.extend(f"  note: {r}" for r in report.get("notes", []))
    return "\n".join(out)


# --------------------------------------------------------------------- run

def run(config, lanes=None, env=None, post=post_json, log=None):
    """Library entry point (the CLI, run_matrix.py and the AWS probe runner
    all call this). Never raises: any failure becomes verdict "error".

    config keys: model, lanes_file, relay_base, relay_key_env, raw_dir
    (None = keep responses in memory only; otherwise each run saves into
    raw_dir/<model>/<UTC timestamp>/), replicates, max_requests, workers,
    deadline_s, analyze_only, run (with analyze_only: the run directory name
    to read; default the latest), request_extra and max_tokens (model-level
    defaults applied to a relay lane added via relay_base). env is the
    mapping key values are read from (defaults to os.environ).
    """
    model = config.get("model", DEFAULT_MODEL)
    try:
        env = os.environ if env is None else env
        model_extra, model_max = config.get("request_extra"), config.get("max_tokens")
        if lanes is None:
            entry = load_model_entry(model, config.get("lanes_file", DEFAULT_LANES_FILE))
            lanes = lanes_of(entry)
            model_extra = entry.get("request_extra") if model_extra is None else model_extra
            model_max = entry.get("max_tokens") if model_max is None else model_max
        if config.get("relay_base"):
            lanes = retarget_relay(lanes, model, config["relay_base"],
                                   config.get("relay_key_env", "DEEPRELAY_API_KEY"),
                                   int(config.get("relay_top_logprobs", 1)), model_extra, model_max)
        validate_lanes(lanes)
        replicates = int(config.get("replicates", DEFAULT_REPLICATES))
        model_dir = config.get("raw_dir")
        if model_dir is not None:
            model_dir = pathlib.Path(model_dir) / model.replace("/", "_")

        if config.get("analyze_only"):
            if model_dir is None:
                raise ValueError("analyze_only needs raw_dir")
            run_dir = (model_dir / config["run"]) if config.get("run") else latest_run_dir(model_dir)
            if run_dir is None or not run_dir.is_dir():
                raise ValueError(f"no saved run under {model_dir}"
                                 + (f" named {config['run']!r}" if config.get("run") else ""))
            responses = load_saved(lanes, run_dir, replicates)
            records, skipped = [], []
            source = {"mode": "analyze_only", "raw_dir": str(run_dir),
                      "run": run_dir.name, "responses_loaded": len(responses)}
        else:
            run_dir = new_run_dir(model_dir) if model_dir is not None else None
            responses, records, skipped = collect(
                lanes, env, run_dir, replicates,
                int(config.get("max_requests", DEFAULT_MAX_REQUESTS)),
                int(config.get("workers", DEFAULT_WORKERS)), post=post, log=log,
                deadline_s=float(config.get("deadline_s", DEFAULT_DEADLINE_S)))
            source = {"mode": "fresh", "raw_dir": str(run_dir) if run_dir else None,
                      "run": run_dir.name if run_dir else None,
                      "http_calls": sum(r.get("http_calls", 0) for r in records),
                      "errors": sum(1 for r in records if r["status"] == "error")}
        report = analyze(lanes, responses, replicates, skipped)
        report["source"] = source
        report["requests"] = {
            "sent": source.get("http_calls", 0),
            "errors": [r for r in records if r["status"] == "error"],
        }
        report["request"] = {"max_tokens": int(model_max or DEFAULT_MAX_TOKENS),
                             "request_extra": dict(model_extra or {})}
    except Exception as e:  # never raise to a caller (Lambda, CLI)
        report = {"verdict": "error", "reasons": [f"{type(e).__name__}: {e}"],
                  "requests": {"sent": 0, "errors": []}}
        responses = {}
    report["model"] = model
    report["harness_version"] = HARNESS_VERSION
    report["replicates"] = config.get("replicates", DEFAULT_REPLICATES)
    try:
        report["text"] = render(report, responses)
    except Exception as e:
        report["text"] = f"VERDICT: {report['verdict'].upper()} (render failed: {e!r})"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lanes-file", default=str(DEFAULT_LANES_FILE),
                    help="matrix.json (or the older lanes file shape)")
    ap.add_argument("--relay-base", help="also probe the model through the relay at this base "
                                         "as an extra lane (the private runner's flow)")
    ap.add_argument("--relay-key-file", help="credentials JSON with api_base + api_key; "
                                             "implies --relay-base unless given")
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    ap.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                    help="ceiling on HTTP calls, retries included (default %(default)s)")
    ap.add_argument("--deadline-s", type=float, default=DEFAULT_DEADLINE_S,
                    help="whole-run deadline; unfinished probes become errors (default %(default)s)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                    help="each run saves its responses in its own timestamped subdirectory "
                         "here; --analyze-only reads one of them")
    ap.add_argument("--analyze-only", action="store_true",
                    help="analyze one saved run (the latest, or --run); send nothing")
    ap.add_argument("--run", help="with --analyze-only: the saved run directory name to read")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    relay_base = args.relay_base
    if args.relay_key_file:
        with open(os.path.expanduser(args.relay_key_file)) as f:
            cred = json.load(f)
        env["DEEPRELAY_API_KEY"] = cred["api_key"]
        relay_base = relay_base or cred.get("api_base")
    elif relay_base is None and env.get("DEEPRELAY_API_KEY") and env.get("DEEPRELAY_API_BASE"):
        relay_base = env["DEEPRELAY_API_BASE"]

    config = {"model": args.model, "lanes_file": args.lanes_file, "relay_base": relay_base,
              "raw_dir": args.raw_dir, "replicates": args.replicates,
              "max_requests": args.max_requests, "deadline_s": args.deadline_s,
              "workers": args.workers, "analyze_only": args.analyze_only, "run": args.run}
    report = run(config, env=env, log=lambda m: print(m, file=sys.stderr, flush=True))
    print(report["text"])
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({k: v for k, v in report.items() if k != "text"}, f, indent=2)
    sys.exit(0 if report["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
