#!/usr/bin/env python3
"""run_matrix: verify every model in matrix.json and write ONE report in
the deeprelay benchmark schema (schema_version 1), the document that
POST /internal/benchmarks/ingest accepts and `deeprelay benchmarks latest`
decodes.

Two tests run per model and both land in the report:

  * the sampling test, Model Equality Testing (Gao, Liang, Guestrin, ICLR
    2025) through the authors' package: model_equality.py. This is the
    judge. A model is verified when the relay's sampled completions sit
    inside the band the reference deployments of the same weights span,
    and a known-different control sits outside it;
  * the top-k logprob fingerprint: model_fingerprint.py. Corroborating
    evidence where a host returns logprobs; it cannot verify a model on
    its own any more, but a logprob MISMATCH still vetoes a pass (logprob
    analysis is the more sensitive instrument where it exists).

    python3 run_matrix.py --json-out report.json --raw-dir raw/ \\
        [--models deeprelay/deepseek-v4.1-flash,deeprelay/glm-5.3] [--summary-out summary.md]

Exit codes (a run that measured nothing must never exit 0):

    0  at least one model was judged (a subject lane reached a verdict) and
       no model is a mismatch
    2  any model is a mismatch
    1  nothing could be judged (every model insufficient or error), or the
       tool itself failed

Provenance comes from the GitHub Actions environment when present
(GITHUB_RUN_ID and friends); a local run gets a local-<timestamp> id.
"""

import argparse
import collections
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.parse

import model_equality as me
import model_fingerprint as mf

SCHEMA_VERSION = 1
EXIT_OK, EXIT_NOTHING_JUDGED, EXIT_MISMATCH = 0, 1, 2


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def rfc3339(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=str(mf.HERE))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def run_meta(env, started):
    """The run block: GitHub Actions provenance when present, else a local
    placeholder so the report is still ingestible."""
    run_id = env.get("GITHUB_RUN_ID") or f"local-{started.strftime('%Y%m%dT%H%M%SZ')}"
    server = env.get("GITHUB_SERVER_URL") or "https://github.com"
    repo = env.get("GITHUB_REPOSITORY") or ""
    url = f"{server}/{repo}/actions/runs/{run_id}" if env.get("GITHUB_RUN_ID") and repo else ""
    try:
        attempt = max(1, int(env.get("GITHUB_RUN_ATTEMPT") or 1))
    except ValueError:
        attempt = 1
    return {
        "id": run_id,
        "url": url,
        "attempt": attempt,
        "repository": repo,
        "workflow": env.get("GITHUB_WORKFLOW") or "",
        "commit_sha": env.get("GITHUB_SHA") or git_sha(),
        "trigger": env.get("GITHUB_EVENT_NAME") or "local",
        "started_at": rfc3339(started),
        "finished_at": rfc3339(started),
        "duration_ms": 0,
    }


def host_of(base_url):
    try:
        return urllib.parse.urlparse(base_url).netloc
    except Exception:
        return ""


def fingerprint_section(report):
    """The per-model fingerprint report as the server stores it: the run()
    report minus the human text, with `requests` as the HTTP call count
    (the wire type is an int) and the per-probe errors under
    `request_errors`."""
    fp = {k: v for k, v in report.items() if k not in ("text", "model", "replicates")}
    reqs = report.get("requests") or {}
    fp["requests"] = int(reqs.get("sent", 0)) if isinstance(reqs, dict) else int(reqs or 0)
    fp["request_errors"] = list(reqs.get("errors", [])) if isinstance(reqs, dict) else []
    fp.setdefault("reasons", [])
    fp.setdefault("notes", [])
    fp.setdefault("lanes", [])
    fp.setdefault("skipped", [])
    fp.setdefault("noise", {})
    fp.setdefault("pairs", [])
    fp.setdefault("lane_verdicts", {})
    fp.setdefault("controls", [])
    fp.setdefault("prompts", list(mf.PROMPTS))
    fp.setdefault("thresholds", {})
    fp["skipped"] = [s["lane"] if isinstance(s, dict) else str(s) for s in fp["skipped"]]
    return fp


def combine(logprob_verdict, samples_verdict):
    """The published verdict. The sampling test judges; the logprob
    fingerprint can only veto. With no sampling test (--no-samples, or the
    package missing) the logprob verdict stands, as before 2026.11.10."""
    if samples_verdict is None:
        return logprob_verdict
    if "mismatch" in (logprob_verdict, samples_verdict):
        return "mismatch"
    return samples_verdict


def model_result(entry, lanes, report, samples=None):
    """One matrix row's outcome in the bench schema. `samples` is the
    model_equality.run() report, or None when the sampling test did not
    run."""
    fp = fingerprint_section(report)
    logprob_verdict = fp.get("verdict", "error")
    if samples is not None:
        sm = {k: v for k, v in samples.items() if k != "text"}
        fp["samples"] = sm
        fp["method"] = "model_equality_testing"
        fp["logprobs_verdict"] = logprob_verdict
        fp["verdict"] = combine(logprob_verdict, sm.get("verdict", "error"))
        fp["reasons"] = list(sm.get("reasons") or []) + [f"logprobs: {r}" for r in fp.get("reasons") or []]
        fp["requests"] = int(fp.get("requests", 0)) + int(sm.get("requests", 0) or 0)
        fp["request_errors"] = list(fp.get("request_errors") or []) + list(sm.get("request_errors") or [])
        judged_lanes = {l["name"] for l in sm.get("lanes", []) if not l.get("dropped")}
        sample_verdicts = sm.get("lane_verdicts") or {}
    else:
        judged_lanes, sample_verdicts = set(), {}
    lane_verdicts = fp.get("lane_verdicts") or {}
    out_lanes = []
    for l in lanes:
        if l["role"] == "control":
            continue
        role = "subject" if l["role"] in mf.VERDICT_ROLES else l["role"]
        if mf.is_text(l) and l["name"] not in judged_lanes:
            # Not an independent deployment the verdict compared against:
            # the API counts "reference" lanes as the deployments a verified
            # model matched. A text lane that took part in the sampling
            # judgement is a reference; one that did not is listed only.
            role = "text_reference"
        if role == "subject":
            lv = lane_verdicts.get(l["name"], {}).get("verdict")
            lp = lv or ("error" if logprob_verdict == "error" else "insufficient")
            if samples is not None:
                sv = sample_verdicts.get(l["name"], {}).get("verdict")
                sv = sv or ("error" if sm.get("verdict") == "error" else "insufficient")
                wv = "mismatch" if "mismatch" in (lp, sv) else sv
            else:
                wv = lp
        else:
            wv = "reference"
        out_lanes.append({"name": l["name"], "role": role, "model": l["model"],
                          "host": host_of(l["base_url"]), "weights_verdict": wv})
    skipped = {s["lane"]: s["reason"] for s in (report.get("skipped_lanes") or [])}
    return {"id": entry["id"], "display_name": entry.get("display_name", entry["id"]),
            "weights": entry.get("weights", ""), "lanes": out_lanes, "fingerprint": fp,
            "skipped": skipped}


def judged(model_result_):
    """A model was judged when its subject lane reached a verdict (same_model,
    mismatch or inconclusive): something was measured and compared."""
    return any(l["role"] == "subject" and l["weights_verdict"] in ("same_model", "mismatch", "inconclusive")
               for l in model_result_["lanes"])


def exit_code(models):
    if any((m.get("fingerprint") or {}).get("verdict") == "mismatch" for m in models):
        return EXIT_MISMATCH
    if any(judged(m) for m in models):
        return EXIT_OK
    return EXIT_NOTHING_JUDGED


def build_report(matrix, results, run, settings):
    """results: [(entry, lanes, logprob report, samples report or None)] in
    matrix order."""
    models = [model_result(entry, lanes, report, samples) for entry, lanes, report, samples in results]
    verdicts, errors = collections.Counter(), collections.Counter()
    requests = 0
    for m in models:
        fp = m["fingerprint"]
        verdicts[fp.get("verdict", "error")] += 1
        requests += fp.get("requests", 0)
        for e in fp.get("request_errors", []):
            errors[e.get("taxonomy", "unknown")] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": mf.HARNESS_VERSION,
        "run": run,
        "settings": settings,
        "models": models,
        "summary": {"models": len(models), "lanes": sum(len(m["lanes"]) for m in models),
                    "requests": requests, "verdicts": dict(verdicts), "errors": dict(errors)},
    }


def settings_block(defaults, replicates, max_tokens, samples=True):
    out = {
        "perf_trials": 0, "perf_max_tokens": 0, "perf_concurrency": 0, "perf_prompts": [],
        "fingerprint_replicates": replicates,
        "fingerprint_max_tokens": max_tokens,
        "fingerprint_prompts": len(mf.PROMPTS),
        "request_timeout_seconds": int(defaults.get("request_timeout_seconds", mf.DEFAULT_TIMEOUT)),
        "skip_perf": True, "skip_fingerprint": False,
    }
    if samples:
        d = (defaults or {}).get("samples") or {}
        out.update({"samples_method": "model_equality_testing",
                    "samples_prompts": len(me.PROMPT_IDS),
                    "samples_per_prompt": int(d.get("per_prompt", me.PER_PROMPT)),
                    "samples_max_tokens": int(d.get("max_tokens", me.MAX_TOKENS)),
                    "samples_permutations": int(d.get("permutations", me.PERMUTATIONS))})
    return out


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def summary_markdown(report):
    lines = [f"## Model-weights fingerprint, harness {report['harness_version']}", "",
             f"Run `{report['run']['id']}` ({report['run']['trigger']}), "
             f"{report['summary']['requests']} requests, verdicts: "
             + ", ".join(f"{k} {v}" for k, v in sorted(report["summary"]["verdicts"].items())), "",
             "| model | verdict | samples (MET) | mmd | band | p | control | logprobs | distance | limit | why |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in report["models"]:
        fp = m["fingerprint"]
        subj = next((l for l in m["lanes"] if l["role"] == "subject"), None)
        lv = (fp.get("lane_verdicts") or {}).get(subj["name"], {}) if subj else {}
        sm = fp.get("samples") or {}
        sv = (sm.get("lane_verdicts") or {}).get(subj["name"], {}) if subj else {}
        ctrl = ", ".join(f"{c['lane']} {'yes' if c['separates'] else 'NO'}" for c in sm.get("controls") or [])
        why = sv.get("reason") or lv.get("reason") or "; ".join(fp.get("reasons") or []) or ""
        lines.append(f"| {m['display_name']} | **{fp.get('verdict', 'error')}** | "
                     f"{sm.get('verdict', 'n/a')} | {_f(sv.get('distance'))} | {_f(sv.get('band'))} | "
                     f"{_f(sv.get('p'), 3)} | {ctrl} | "
                     f"{fp.get('logprobs_verdict', fp.get('verdict', 'error')) if sm else fp.get('verdict', 'error')} | "
                     f"{_f(lv.get('distance'))} | {_f(lv.get('limit'))} | {why.replace('|', '/')} |")
        extra = [r for r in fp.get("reasons") or [] if r != why]
        for r in extra:
            lines.append(f"|  | | | | | | | | | | {r.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv=None, env=None, run_model=None, run_samples=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(mf.DEFAULT_LANES_FILE))
    ap.add_argument("--models", help="comma-separated deeprelay model ids (default: every model)")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--summary-out", help="write a Markdown summary here (for GITHUB_STEP_SUMMARY)")
    ap.add_argument("--raw-dir", help="save raw responses under here, one run directory per model")
    ap.add_argument("--replicates", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--deadline-s", type=float, help="per-model deadline")
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-judge the latest saved run of each model under --raw-dir; send nothing")
    ap.add_argument("--quiet", action="store_true", help="do not print the per-model text reports")
    ap.add_argument("--no-samples", action="store_true",
                    help="skip the sampling test (Model Equality Testing); the logprob fingerprint alone judges")
    args = ap.parse_args(argv)
    env = dict(os.environ) if env is None else env
    run_model = run_model or mf.run
    run_samples = run_samples or me.run
    log = (lambda m: print(m, file=sys.stderr, flush=True))

    matrix = mf.load_matrix(args.matrix)
    defaults = matrix.get("defaults") or {}
    entries = list(mf.model_entries(matrix).values())
    if args.analyze_only and not args.raw_dir:
        log("--analyze-only needs --raw-dir")
        return EXIT_NOTHING_JUDGED
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        known = {e["id"] for e in entries}
        unknown = [m for m in want if m not in known]
        if unknown:
            log(f"unknown model id(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
            return EXIT_NOTHING_JUDGED
        entries = [e for e in entries if e["id"] in want]
    replicates = args.replicates or int(defaults.get("replicates", mf.DEFAULT_REPLICATES))
    workers = args.workers or int(defaults.get("workers", mf.DEFAULT_WORKERS))
    max_requests = args.max_requests or int(defaults.get("max_requests", mf.DEFAULT_MAX_REQUESTS))
    deadline_s = args.deadline_s or float(defaults.get("deadline_s", mf.DEFAULT_DEADLINE_S))

    started = utc_now()
    t0 = time.monotonic()
    run = run_meta(env, started)
    results = []
    for entry in entries:
        lanes = mf.lanes_of(entry)
        log(f"=== {entry['id']} ({len(lanes)} lanes) ===")
        report = run_model({"model": entry["id"], "raw_dir": args.raw_dir, "replicates": replicates,
                            "workers": workers, "max_requests": max_requests, "deadline_s": deadline_s,
                            "request_extra": entry.get("request_extra"),
                            "max_tokens": entry.get("max_tokens"),
                            "analyze_only": bool(args.analyze_only)},
                           lanes=lanes, env=env, log=log)
        samples = None
        sset = me.settings_for(entry, defaults)
        if not args.no_samples and sset.get("enabled", True):
            run_dir = (report.get("source") or {}).get("raw_dir")
            # The sampling sweep has its own budget and deadline (matrix
            # defaults.samples / the model's samples block): its request
            # count is an order of magnitude above the logprob probes'.
            samples = run_samples(entry, lanes, env, sset, raw_dir=run_dir, log=log,
                                  analyze_only=bool(args.analyze_only))
        results.append((entry, lanes, report, samples))
        if not args.quiet:
            print(report.get("text", ""), flush=True)
            if samples is not None:
                print(samples.get("text", ""), flush=True)
            print(flush=True)
    finished = utc_now()
    run["finished_at"] = rfc3339(finished)
    run["duration_ms"] = int((time.monotonic() - t0) * 1000)
    settings = settings_block(defaults, replicates,
                              max((int(e.get("max_tokens") or defaults.get("max_tokens") or mf.DEFAULT_MAX_TOKENS)
                                   for e in entries), default=mf.DEFAULT_MAX_TOKENS),
                              samples=not args.no_samples)
    report = build_report(matrix, results, run, settings)
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=1)
    md = summary_markdown(report)
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            f.write(md)
    print(md, flush=True)
    code = exit_code(report["models"])
    log(f"exit {code}: " + {EXIT_OK: "judged, no mismatch", EXIT_MISMATCH: "MISMATCH",
                            EXIT_NOTHING_JUDGED: "nothing could be judged"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
