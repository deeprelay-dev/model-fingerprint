#!/usr/bin/env python3
"""model_equality: the sampling test. Are the completions api.deeprelay.ai
returns for a model drawn from the same distribution as the completions
independent deployments of the published weights return?

The test is Model Equality Testing (Gao, Liang and Guestrin, "Model
Equality Testing: Which Model Is This API Serving?", ICLR 2025,
https://arxiv.org/abs/2410.20247), run through the authors' package
exactly as published:

    pip install model-equality-testing      https://github.com/i-gao/model-equality-testing

Every number this module reports comes out of that package:
`CompletionSample` holds the samples, `mmd_hamming` is the statistic
(maximum mean discrepancy with the Hamming kernel over unicode
codepoints, completions truncated or padded to L characters) and
`run_two_sample_test(..., pvalue_type="permutation_pvalue")` is the
p-value. Nothing statistical is reimplemented here; this file only
collects samples from the lanes of matrix.json and applies a decision
rule to the package's output. A reader with their own vendor keys can
rerun the whole thing and check every pair.

How it works
  * Every lane (the subject through the relay, each reference deployment
    of the same weights, each known-different control) answers the same
    SAMPLE_PROMPTS at temperature 1, top_p 1, PER_PROMPT times. Sampling
    at temperature 1 is what the test needs: it compares distributions,
    not greedy answers, so no logprobs are required. That is why a
    vendor API that returns no logprobs (Z.ai, Moonshot) is a full
    reference here, where the logprob fingerprint could only list it.
  * For every pair of lanes the package computes the MMD statistic, and
    for the pairs that decide the verdict (subject vs reference, subject
    vs control) the permutation p-value as well.
  * Same weights on different serving stacks (a different quantisation,
    sampler, chat template or thinking budget) do NOT give identical
    distributions, and the test is sensitive enough to say so: the
    reference deployments usually reject each other at p < 0.05. The
    verdict is therefore calibrated the way the logprob fingerprint is:
    the distances between the reference deployments set the BAND of
    "same weights, different host"; a subject lane must match its nearest
    reference (statistically indistinguishable, p >= ALPHA, or inside the
    band) and be closer to every reference than to any control; and at
    least one control, a different model, must sit outside the band with
    p < ALPHA against the subject and outside the band against every
    reference. A control that fails to separate makes the run
    inconclusive: the test has not been shown to discriminate on that
    day's samples.
  * A subject lane is a MISMATCH only when it is outside the band
    against every reference AND at least as far from the nearest
    reference as it is from the nearest control: it looks like a
    different model, not like a noisy same-weights host.

Prompt design matters more than anything else here. A prompt the model
has memorised (a famous article opening) gives every host the same
near-deterministic text, and a one-token difference in which memorised
phrasing a host's sampler favours becomes a large position-sensitive
Hamming distance between two deployments of the same weights.
SAMPLE_PROMPTS therefore asks for continuations of passages written for
this harness, in several languages, plus short open writing, code and
explanation tasks, and asks the model not to repeat the passage (whether
a host's chat template makes the model restate the prompt is a template
difference, not a weights difference).
"""

import collections
import concurrent.futures
import json
import os
import pathlib
import threading
import time
import urllib.error

import model_fingerprint as mf

PAPER = "Gao, Liang, Guestrin. Model Equality Testing: Which Model Is This API Serving? ICLR 2025"
PAPER_URL = "https://arxiv.org/abs/2410.20247"
PACKAGE = "model-equality-testing"
PACKAGE_URL = "https://github.com/i-gao/model-equality-testing"
STATISTIC = "mmd_hamming"
PVALUE = "permutation_pvalue"

# Defaults; matrix.json "defaults.samples" and a model's "samples" block
# override them.
PER_PROMPT = 15          # samples per prompt per lane
MAX_TOKENS = 1200        # thinking models spend most of this on reasoning
L = 200                  # characters compared (the package's README setting)
PERMUTATIONS = 100       # permutation p-value resolution 0.01
ALPHA = 0.05             # the paper's rejection level
WORKERS = 6              # concurrent requests per lane (lanes run in parallel)
DEADLINE_S = 900.0       # the sampling sweep's own deadline per model
MIN_LANE_FRACTION = 0.8  # a lane that returned less than this of its plan is dropped
MIN_PER_PROMPT = 5       # a prompt with fewer samples in a lane drops the lane
MIN_REFERENCES = 2       # a band needs at least one reference pair
REQUEST_TIMEOUT = 120.0

CONTINUE = ("Continue the passage below in the same register. Do not repeat or "
            "summarise it; begin directly with the next sentence.\n\n")
CONTINUE_ES = ("Continúa el siguiente pasaje con el mismo tono. No lo repitas ni lo "
               "resumas; empieza directamente con la siguiente frase.\n\n")
CONTINUE_FR = ("Poursuis le passage ci-dessous dans le même registre. Ne le répète pas "
               "et ne le résume pas ; commence directement par la phrase suivante.\n\n")
CONTINUE_DE = ("Setze den folgenden Abschnitt im gleichen Stil fort. Wiederhole oder "
               "fasse ihn nicht zusammen; beginne direkt mit dem nächsten Satz.\n\n")
CONTINUE_RU = ("Продолжи приведённый ниже отрывок в том же стиле. Не повторяй и не "
               "пересказывай его; начни сразу со следующего предложения.\n\n")
CONTINUE_ZH = "请以相同的风格续写下面这段文字。不要重复或概括原文，直接从下一句开始。\n\n"

SAMPLE_PROMPTS = {
    "s01_harbour": CONTINUE + (
        "The harbour town of Ebbsmere grew up around a single stone jetty built by fishermen "
        "who had tired of hauling their boats over the shingle. By the time the railway "
        "arrived, the jetty had become a quay, the quay had sprouted warehouses, and the "
        "warehouses had"),
    "s02_clockmaker": CONTINUE + (
        "Every clockmaker in the valley learned the trade from the same two brothers, and "
        "for a century their habits could be read in the movements: a particular way of "
        "cutting the escapement, a preference for brass over steel in the smaller wheels. "
        "The last apprentice they trained"),
    "s03_lichen": CONTINUE + (
        "Lichens on the north face of the monument grow at a rate that depends less on "
        "rainfall than on how often the stone is cleaned. Surveys taken a decade apart "
        "showed"),
    "s04_glasshouse": CONTINUE + (
        "The glasshouse was never intended to be permanent. It had been raised in a hurry, "
        "the winter the orchard failed, so that the cuttings would survive until spring. "
        "Forty years later"),
    "s05_canal_es": CONTINUE_ES + (
        "El canal que une las dos lagunas fue excavado a mano por los vecinos de tres "
        "pueblos, que se turnaban por semanas. Cuando por fin corrió el agua,"),
    "s06_library_fr": CONTINUE_FR + (
        "La bibliothèque du village occupait l'ancienne gare, et les horaires des trains "
        "restaient peints au-dessus du comptoir de prêt. Les lecteurs les plus âgés"),
    "s07_mill_de": CONTINUE_DE + (
        "Die Mühle am Oberlauf des Baches mahlte zuletzt im Herbst vor dem großen Hochwasser. "
        "Als das Wasser zurückging,"),
    "s08_station_ru": CONTINUE_RU + (
        "Метеостанция на перевале работала круглый год, и зимой единственной дорогой к ней "
        "была лыжня, которую наблюдатель прокладывал сам. Каждое утро он"),
    "s09_teahouse_zh": CONTINUE_ZH + (
        "山脚下的茶馆开了六十多年，招牌早已褪色，但每天清晨仍有老人来喝第一壶茶。茶馆的主人"),
    "s10_dawn": "Write one paragraph describing a port city in the hour before dawn. "
               "Begin immediately with the description.",
    "s11_gadget": "Write a short product description, three or four sentences, for a "
                  "pocket-sized device that measures the ripeness of fruit by tapping it.",
    "s12_dialogue": "Write a six-line dialogue between a lighthouse keeper and a visiting "
                    "inspector who has arrived a day early. Dialogue only.",
    "s13_fib_py": "Write a Python function that returns the n-th Fibonacci number. "
                  "Code only, no explanation.",
    "s14_date_js": "Write a JavaScript function that parses a date string in the form "
                   "DD/MM/YYYY and returns a Date object, or null if the string is invalid. "
                   "Code only, no explanation.",
    "s15_bread": "Explain in two or three sentences why bread dough rises.",
    "s16_tides": "Explain in two or three sentences why there are two high tides a day.",
}
PROMPT_IDS = list(SAMPLE_PROMPTS)


# ------------------------------------------------------------------ config

def settings_for(entry, defaults):
    """The sampling settings of one matrix entry: module defaults, then
    matrix defaults.samples, then the entry's samples block. request_extra
    is the entry's model-level request_extra unless the samples block
    carries its own (a thinking-only model needs reasoning_effort low here
    while the logprob lanes think at the default)."""
    s = {"per_prompt": PER_PROMPT, "max_tokens": MAX_TOKENS, "L": L,
         "permutations": PERMUTATIONS, "alpha": ALPHA, "workers": WORKERS,
         "deadline_s": DEADLINE_S, "max_requests": None,
         "request_extra": dict(entry.get("request_extra") or {}), "enabled": True}
    for block in ((defaults or {}).get("samples") or {}, entry.get("samples") or {}):
        for k, v in block.items():
            if k == "request_extra":
                s["request_extra"] = dict(v or {})
            else:
                s[k] = v
    return s


def build_body(lane, prompt, max_tokens, request_extra, lane_extra=None):
    """The sampling request. `request_extra` is the entry's sampling
    request_extra (settings_for), `lane_extra` the keys this lane's own
    request_extra sets beyond the entry's (lane_overrides), and the lane's
    optional samples_extra wins over both. A lane marked stream is sampled
    over SSE, as its logprob probes are (Together's Qwen endpoints accept
    nothing else)."""
    body = {"model": lane["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens), "temperature": 1, "top_p": 1,
            "stream": bool(lane.get("stream"))}
    body.update(request_extra or {})
    body.update(lane_extra or {})
    body.update(lane.get("samples_extra") or {})
    return body


def lane_overrides(lane, entry):
    """The request_extra keys a lane sets itself. lanes_of() folds the
    entry's request_extra into every lane, so a lane's own switch (say
    reasoning_effort none on one control) is whatever differs from the
    entry's value."""
    base = entry.get("request_extra") or {}
    return {k: v for k, v in (lane.get("request_extra") or {}).items() if base.get(k) != v}


def content_of(payload):
    try:
        msg = payload["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return msg.get("content")


# ------------------------------------------------------------------ sampling

def sample_path(run_dir, lane, pid, rep):
    return pathlib.Path(run_dir) / "samples" / lane["name"] / f"{pid}-{rep}.json"


def call(lane, key, pid, rep, run_dir, max_tokens, request_extra, post=mf.post_json,
         budget=None, deadline=None, lane_extra=None):
    """One sample. Returns (text_or_None, record); never raises. Retries a
    429, 5xx or transport error up to mf.MAX_RETRIES times."""
    base = {"lane": lane["name"], "prompt": pid, "rep": rep, "http_calls": 0, "retries": 0}
    url = lane["base_url"].rstrip("/") + "/chat/completions"
    body = build_body(lane, SAMPLE_PROMPTS[pid], max_tokens, request_extra, lane_extra)
    err = None
    while True:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            err = err or {"taxonomy": "deadline", "error": "run deadline reached before this sample"}
            break
        if budget is not None and not budget.take():
            err = err or {"taxonomy": "max_requests", "error": "--max-requests exhausted"}
            break
        timeout = REQUEST_TIMEOUT if deadline is None else max(1.0, min(REQUEST_TIMEOUT, deadline - now))
        base["http_calls"] += 1
        payload = None
        try:
            payload = post(url, key, body, timeout=timeout)
        except urllib.error.HTTPError as e:
            err = {"taxonomy": mf.classify_exception(e),
                   "error": f"HTTP {e.code}: {mf.http_error_body(e)[:300]}"}
        except Exception as e:
            err = {"taxonomy": mf.classify_exception(e), "error": repr(e)[:300]}
        if payload is not None:
            text = content_of(payload)
            if isinstance(text, str) and text.strip():
                rec = dict(base, status="ok", chars=len(text),
                           finish=(payload.get("choices") or [{}])[0].get("finish_reason"))
                if run_dir is not None:
                    try:
                        mf.write_atomic(sample_path(run_dir, lane, pid, rep), payload)
                    except OSError as e:
                        rec["save_error"] = repr(e)[:200]
                return text, rec
            err = {"taxonomy": "empty_content",
                   "error": "no content in the reply (reasoning exhausted max_tokens?)"}
        if base["retries"] >= mf.MAX_RETRIES or err["taxonomy"] not in mf.RETRYABLE + ("empty_content",):
            break
        base["retries"] += 1
        pause = mf.RETRY_BACKOFF_S * base["retries"]
        if deadline is not None:
            pause = min(pause, max(0.0, deadline - time.monotonic()))
        time.sleep(pause)
    return None, dict(base, status="error", **(err or {}))


def collect(lanes, env, run_dir, settings, max_requests, post=mf.post_json, log=None,
            deadline_s=mf.DEFAULT_DEADLINE_S, entry=None):
    """Sample every lane. Returns (samples, records, skipped): samples maps
    (lane_name, pid, rep) -> text. Lanes run concurrently, each with its
    own worker pool, so one slow vendor does not hold the others."""
    log = log or (lambda m: None)
    entry = entry or {}
    runnable, skipped = mf.partition_lanes(lanes, env)
    per_prompt = int(settings["per_prompt"])
    budget = mf.Budget(max_requests)
    deadline = time.monotonic() + float(deadline_s)
    samples, records = {}, []
    lock = threading.Lock()

    def one(lane, pid, rep):
        text, rec = call(lane, env[lane["key_env"]], pid, rep, run_dir, settings["max_tokens"],
                         settings["request_extra"], post=post, budget=budget, deadline=deadline,
                         lane_extra=lane_overrides(lane, entry))
        with lock:
            records.append(rec)
            if text is not None:
                samples[(lane["name"], pid, rep)] = text

    def run_lane(lane):
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(settings["workers"])) as ex:
            futs = [ex.submit(one, lane, pid, rep) for pid in PROMPT_IDS for rep in range(per_prompt)]
            for f in futs:
                f.result()

    log(f"SAMPLES: {len(runnable)} lanes x {len(PROMPT_IDS)} prompts x {per_prompt} at temperature 1, "
        f"max_tokens {settings['max_tokens']}, deadline {int(deadline_s)}s")
    if runnable:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(runnable)) as ex:
            for f in [ex.submit(run_lane, l) for l in runnable]:
                f.result()
    for lane in runnable:
        n = sum(1 for k in samples if k[0] == lane["name"])
        e = sum(1 for r in records if r["lane"] == lane["name"] and r["status"] == "error")
        log(f"SAMPLES: {lane['name']}: {n} samples, {e} errors")
    return samples, records, skipped


def load_saved(lanes, run_dir, per_prompt):
    samples = {}
    for lane in lanes:
        for pid in PROMPT_IDS:
            for rep in range(per_prompt):
                p = sample_path(run_dir, lane, pid, rep)
                if p.is_file():
                    try:
                        with open(p) as f:
                            text = content_of(json.load(f))
                    except (OSError, ValueError):
                        continue
                    if isinstance(text, str) and text.strip():
                        samples[(lane["name"], pid, rep)] = text
    return samples


# ------------------------------------------------------------------ judging

def _package():
    """The published package, imported lazily so the logprob fingerprint
    and its tests need no torch."""
    import numpy as np  # noqa: F401
    from model_equality_testing.algorithm import run_two_sample_test
    from model_equality_testing.distribution import CompletionSample
    from model_equality_testing.tests import IMPLEMENTED_TESTS
    try:
        from importlib.metadata import version
        ver = version(PACKAGE)
    except Exception:
        ver = "unknown"
    return run_two_sample_test, CompletionSample, IMPLEMENTED_TESTS[STATISTIC], ver


def completion_sample(CompletionSample, texts, L):
    """texts: list of (prompt index, text) -> the package's sample object
    in unicode codepoint space, padded with -1 to L (its README recipe)."""
    import numpy as np
    prompts = np.array([pi for pi, _ in texts], dtype=np.int64)
    comps = np.full((len(texts), L), -1, dtype=np.int64)
    for i, (_, t) in enumerate(texts):
        cps = [ord(c) for c in t[:L]]
        comps[i, :len(cps)] = cps
    return CompletionSample(prompts=prompts, completions=comps, m=len(PROMPT_IDS))


def lane_texts(samples, name):
    idx = {pid: i for i, pid in enumerate(PROMPT_IDS)}
    return sorted(((idx[pid], text) for (n, pid, rep), text in samples.items() if n == name),
                  key=lambda x: x[0])


def pair_kind(a, b):
    ra, rb = a["role"], b["role"]
    subj = {r for r in (ra, rb) if r in mf.VERDICT_ROLES}
    if "control" in (ra, rb):
        return "subject-control" if subj else "reference-control"
    if subj:
        return "subject-reference" if len(subj) == 1 else "subject-subject"
    return "reference-reference"


def judge(lanes, samples, settings, skipped=(), seed=0, log=None):
    """Apply the decision rule to the package's statistics. Pure given the
    samples (the permutation p-value is seeded)."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_test, CompletionSample, stat_fn, pkg_version = _package()
    Lc, nperm, alpha = int(settings["L"]), int(settings["permutations"]), float(settings["alpha"])
    per_prompt = int(settings["per_prompt"])
    planned = per_prompt * len(PROMPT_IDS)
    by = {l["name"]: l for l in lanes}
    report = {"method": "model_equality_testing", "paper": PAPER, "paper_url": PAPER_URL,
              "package": f"{PACKAGE} {pkg_version}", "package_url": PACKAGE_URL,
              "statistic": STATISTIC, "pvalue": PVALUE, "permutations": nperm, "L": Lc,
              "alpha": alpha, "temperature": 1, "prompts": list(PROMPT_IDS),
              "per_prompt": per_prompt, "lanes": [], "dropped": [], "pairs": [],
              "band": None, "controls": [], "lane_verdicts": {}, "reasons": [], "notes": [],
              "skipped": [s["lane"] if isinstance(s, dict) else str(s) for s in skipped]}

    # Lane sufficiency.
    usable, texts = [], {}
    for l in lanes:
        if l["name"] in report["skipped"]:
            continue
        t = lane_texts(samples, l["name"])
        per = collections.Counter(pi for pi, _ in t)
        thin = [PROMPT_IDS[i] for i in range(len(PROMPT_IDS)) if per.get(i, 0) < MIN_PER_PROMPT]
        row = {"name": l["name"], "role": l["role"], "model": l["model"], "samples": len(t),
               "planned": planned}
        report["lanes"].append(row)
        if len(t) < MIN_LANE_FRACTION * planned or thin:
            why = (f"{len(t)}/{planned} samples" if len(t) < MIN_LANE_FRACTION * planned
                   else f"under {MIN_PER_PROMPT} samples on {', '.join(thin)}")
            report["dropped"].append({"lane": l["name"], "reason": why})
            row["dropped"] = why
            continue
        usable.append(l)
        texts[l["name"]] = t
    subjects = [l for l in usable if l["role"] in mf.VERDICT_ROLES]
    references = [l for l in usable if l["role"] == "reference"]
    controls = [l for l in usable if l["role"] == "control"]

    if not subjects:
        report["verdict"] = "insufficient"
        report["reasons"].append("no subject lane returned enough samples")
        return report
    if len(references) < MIN_REFERENCES:
        report["verdict"] = "insufficient"
        report["reasons"].append(f"{len(references)} usable reference deployment(s); the band needs "
                                 f"at least {MIN_REFERENCES}")
        return report

    cs = {n: completion_sample(CompletionSample, t, Lc) for n, t in texts.items()}
    pairs = {}

    def pair(a, b, with_p):
        key = (a["name"], b["name"])
        if key in pairs:
            return pairs[key]
        if with_p:
            p, stat = run_test(cs[a["name"]], cs[b["name"]], stat_type=STATISTIC,
                               pvalue_type=PVALUE, b=nperm)
            p = float(p)
        else:
            stat, p = stat_fn(cs[a["name"]], cs[b["name"]]), None
        row = {"a": a["name"], "b": b["name"], "kind": pair_kind(a, b), "mmd": float(stat), "p": p}
        pairs[key] = row
        report["pairs"].append(row)
        return row

    # The band: same weights, different hosts.
    ref_pairs = [pair(references[i], references[j], False)
                 for i in range(len(references)) for j in range(i + 1, len(references))]
    band = max(r["mmd"] for r in ref_pairs)
    widest = max(ref_pairs, key=lambda r: r["mmd"])
    report["band"] = {"mmd": band, "pairs": len(ref_pairs), "widest": [widest["a"], widest["b"]],
                      "references": [r["name"] for r in references]}

    # Controls: a different model must sit outside the band against the
    # references (statistic) and against every subject (p-value).
    for c in controls:
        ref_d = [pair(c, r, False)["mmd"] for r in references]
        subj = {s["name"]: pair(s, c, True) for s in subjects}
        far_from_refs = min(ref_d) > band
        far_from_subjects = all(v["mmd"] > band and v["p"] is not None and v["p"] < alpha
                                for v in subj.values())
        report["controls"].append({
            "lane": c["name"], "model": c["model"], "advisory": bool(c.get("advisory")),
            "min_reference_mmd": min(ref_d), "band": band,
            "subjects": {n: {"mmd": v["mmd"], "p": v["p"]} for n, v in subj.items()},
            "separates": bool(far_from_refs and far_from_subjects)})
    gating = [c for c in report["controls"] if not c["advisory"]]
    for c in report["controls"]:
        if not c["separates"]:
            msg = (f"control {c['lane']} did not separate (min mmd to a reference "
                   f"{c['min_reference_mmd']:.4f} vs band {band:.4f}"
                   + "".join(f", p={v['p']:.3f} vs {n}" for n, v in c["subjects"].items()
                             if v["p"] is not None) + ")")
            (report["notes"] if c["advisory"] else report["reasons"]).append(
                msg + ("; advisory, the verdict does not depend on it" if c["advisory"] else ""))
    if not gating:
        report["reasons"].append("no gating control: the test has not been shown to "
                                 "discriminate on these samples")
    elif not any(c["separates"] for c in gating):
        report["reasons"].append("no gating control separates: the test has not been shown "
                                 "to discriminate on these samples")

    # Subjects. A subject is the same model when it matches its NEAREST
    # reference deployment (indistinguishable at alpha, or within the band
    # the reference deployments span among themselves) and is closer to
    # every reference than to any control. Matching every reference is not
    # required: the hosts differ among themselves, and the band is the max
    # over a handful of pairs, so a subject that reproduces one host's
    # distribution can sit just outside the band against another host
    # while still being far closer to it than a different model is.
    for s in subjects:
        rows = {r["name"]: pair(s, r, True) for r in references}
        within = {n: bool(v["p"] >= alpha or v["mmd"] <= band) for n, v in rows.items()}
        nearest = min(rows.values(), key=lambda v: v["mmd"])
        farthest = max(rows.values(), key=lambda v: v["mmd"])
        # Only gating controls guard the verdict; an advisory control is
        # reported but can neither block nor rescue a lane.
        ctrl_d = [c["subjects"][s["name"]]["mmd"] for c in report["controls"] if not c["advisory"]]
        nearest_ok = within[nearest["b"]]
        closer_than_controls = (not ctrl_d) or farthest["mmd"] < min(ctrl_d)
        lv = {"nearest_reference": nearest["b"], "distance": nearest["mmd"], "p": nearest["p"],
              "band": band, "alpha": alpha,
              "references": {n: {"mmd": v["mmd"], "p": v["p"], "within": within[n]} for n, v in rows.items()},
              "farthest_reference": farthest["b"], "farthest_distance": farthest["mmd"],
              "nearest_control": min(ctrl_d) if ctrl_d else None}
        where = (f"nearest {nearest['b']}: mmd {nearest['mmd']:.4f}, p={nearest['p']:.3f}; "
                 f"band {band:.4f}")
        if nearest_ok and closer_than_controls:
            lv["verdict"] = "same_model"
            lv["reason"] = (f"matches the nearest reference ({where}) and is closer to every reference "
                            f"than to any control (farthest {farthest['b']} {farthest['mmd']:.4f} vs "
                            f"nearest control {min(ctrl_d):.4f})" if ctrl_d else
                            f"matches the nearest reference ({where})")
        elif not nearest_ok and ctrl_d and nearest["mmd"] >= min(ctrl_d):
            lv["verdict"] = "mismatch"
            lv["reason"] = (f"outside the band against every reference ({where}) and at least as far "
                            f"from the nearest reference as from the nearest control ({min(ctrl_d):.4f})")
        elif not nearest_ok:
            lv["verdict"] = "inconclusive"
            lv["reason"] = (f"outside the band against every reference ({where}) but closer to the "
                            f"references than to any control ({min(ctrl_d):.4f})" if ctrl_d else
                            f"outside the band against every reference ({where}); no control to compare")
        else:
            lv["verdict"] = "inconclusive"
            lv["reason"] = (f"matches the nearest reference ({where}) but is as far from "
                            f"{farthest['b']} ({farthest['mmd']:.4f}) as from the nearest control "
                            f"({min(ctrl_d):.4f})")
        report["lane_verdicts"][s["name"]] = lv

    verdicts = [v["verdict"] for v in report["lane_verdicts"].values()]
    if "mismatch" in verdicts:
        report["verdict"] = "mismatch"
    elif report["reasons"] or any(v != "same_model" for v in verdicts):
        report["verdict"] = "inconclusive"
    else:
        report["verdict"] = "pass"
    return report


# ------------------------------------------------------------------ render

def render(report):
    out = []
    f = mf._f
    out.append(f"SAMPLES ({report.get('method', '?')}, {report.get('package', '?')}): "
               f"verdict {str(report.get('verdict', '?')).upper()}")
    if report.get("prompts"):
        out.append(f"  {len(report['prompts'])} prompts x {report.get('per_prompt')} samples per lane "
                   f"at temperature 1; {report.get('statistic')} over the first {report.get('L')} "
                   f"characters; {report.get('pvalue')} with b={report.get('permutations')}; "
                   f"alpha {report.get('alpha')}")
    for l in report.get("lanes", []):
        out.append(f"  lane {l['name']:<18} {l['role']:<10} {l['samples']}/{l['planned']} samples"
                   + (f"  DROPPED: {l['dropped']}" if l.get("dropped") else ""))
    band = report.get("band")
    if band:
        out.append(f"  band (same weights, different hosts): mmd {f(band['mmd'])} over "
                   f"{band['pairs']} reference pair(s), widest {band['widest'][0]} vs {band['widest'][1]}")
    for c in report.get("controls", []):
        subj = ", ".join(f"{n}: mmd {f(v['mmd'])} p={f(v['p'], 3)}" for n, v in c["subjects"].items())
        out.append(f"  control {c['lane']:<16} {'separates' if c['separates'] else 'DOES NOT separate':<18}"
                   f" min mmd to a reference {f(c['min_reference_mmd'])}; vs {subj}"
                   + ("  (advisory)" if c.get("advisory") else ""))
    for n, v in report.get("lane_verdicts", {}).items():
        out.append(f"  {n}: {v['verdict'].upper()}: {v['reason']}")
        for r, d in v["references"].items():
            out.append(f"      vs {r:<16} mmd {f(d['mmd'])}  p={f(d['p'], 3)}  "
                       f"{'within' if d['within'] else 'OUTSIDE'}")
    for r in report.get("reasons", []):
        out.append(f"  reason: {r}")
    for n in report.get("notes", []):
        out.append(f"  note: {n}")
    return "\n".join(out)


# ------------------------------------------------------------------ entry

def run(entry, lanes, env, settings, raw_dir=None, max_requests=None, deadline_s=None,
        post=mf.post_json, log=None, analyze_only=False, run_name=None, seed=0):
    """Sample and judge one matrix entry. Never raises: a failure becomes
    verdict "error". raw_dir is the model's run directory (the same one the
    logprob fingerprint saved into) or None. The HTTP budget defaults to
    twice the planned samples (room for retries) and the deadline to the
    settings' deadline_s; both can be overridden per call."""
    max_requests = max_requests or settings.get("max_requests")
    deadline_s = deadline_s or settings.get("deadline_s") or DEADLINE_S
    log = log or (lambda m: None)
    try:
        run_dir = pathlib.Path(raw_dir) if raw_dir else None
        if analyze_only:
            if run_dir is None:
                raise ValueError("analyze_only needs raw_dir")
            samples = load_saved(lanes, run_dir, int(settings["per_prompt"]))
            records, skipped = [], []
            source = {"mode": "analyze_only", "samples_loaded": len(samples)}
        else:
            samples, records, skipped = collect(
                lanes, env, run_dir, settings,
                int(max_requests or len(lanes) * len(PROMPT_IDS) * int(settings["per_prompt"]) * 2),
                post=post, log=log, deadline_s=float(deadline_s), entry=entry)
            source = {"mode": "fresh", "http_calls": sum(r.get("http_calls", 0) for r in records),
                      "errors": sum(1 for r in records if r["status"] == "error")}
        report = judge(lanes, samples, settings, skipped, seed=seed, log=log)
        report["source"] = source
        report["request_errors"] = [r for r in records if r["status"] == "error"]
        report["requests"] = source.get("http_calls", 0)
        report["request"] = {"max_tokens": int(settings["max_tokens"]),
                             "request_extra": dict(settings["request_extra"] or {})}
    except Exception as e:
        report = {"method": "model_equality_testing", "verdict": "error",
                  "reasons": [f"{type(e).__name__}: {e}"], "lane_verdicts": {}, "lanes": [],
                  "pairs": [], "controls": [], "dropped": [], "notes": [], "requests": 0,
                  "request_errors": []}
    report["text"] = render(report)
    return report
