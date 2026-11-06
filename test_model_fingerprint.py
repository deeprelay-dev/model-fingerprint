"""Hermetic tests for model_fingerprint.py: the metric math, both logprob wire
shapes, the content slicer, streaming aggregation, the relative signature,
width-aware distances, the verdict logic, the coverage guards and the request
accounting. No network.

Run: python3 -m unittest discover -s . -p 'test_*.py' -v
"""

import io
import json
import math
import pathlib
import sys
import tempfile
import time
import unittest
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_fingerprint as mf  # noqa: E402

# Retries are real but instant under test.
mf.RETRY_BACKOFF_S = 0.0

P = list(mf.PROMPTS)


def content_shape(tokens, prompt_tokens=40, k=5, content=None, spread=0.0, reasoning=None,
                  tail=(), chosen=-0.1):
    """OpenAI logprobs.content[] response. Each position's top-k is the
    chosen token at logprob -0.1 plus k-1 alternatives; `spread` shifts the
    alternatives so two responses can be made measurably different.
    `content` defaults to the concatenated tokens; `reasoning` prepends
    reasoning tokens to the logprob stream; `tail` appends stop tokens."""
    rows = []
    for t in list(reasoning or []) + list(tokens) + list(tail):
        top = [{"token": t, "logprob": chosen}]
        for i in range(1, k):
            top.append({"token": f"{t}_alt{i}", "logprob": -2.0 - i - spread * i})
        rows.append({"token": t, "logprob": chosen, "top_logprobs": top if k > 1 else []})
    msg = {"content": "".join(tokens) if content is None else content}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    return {"choices": [{"message": msg, "logprobs": {"content": rows}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(rows)}}


def dist_shape(dist, sig=10, logprobs=True, first=None):
    """One-position response from an explicit distribution. `first` forces
    the emitted token."""
    tok = first or max(dist, key=dist.get)
    row = {"token": tok, "logprob": math.log(dist.get(tok, 1e-3)),
           "top_logprobs": [{"token": t, "logprob": math.log(v)} for t, v in dist.items()]}
    return {"choices": [{"message": {"content": tok},
                         "logprobs": {"content": [row]} if logprobs else None}],
            "usage": {"prompt_tokens": sig}}


def no_logprobs(prompt_tokens=40, content="x"):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3}}


def lane(name, role, key_env="K", k=5, **extra):
    return dict({"name": name, "role": role, "base_url": f"https://{name}/v1", "model": name,
                 "key_env": key_env, "top_logprobs": k}, **extra)


LANES = [lane("subj", "subject", "S_KEY"), lane("refA", "reference", "A_KEY"),
         lane("refB", "reference", "B_KEY"), lane("ctrl", "control", "A_KEY")]

REF_A = {"x": .90, "y": .10}
REF_B = {"x": .88, "y": .12}
FAR = {"x": .55, "y": .45}


def fill(responses, name, make, reps=2):
    for i, pid in enumerate(P):
        for rep in range(reps):
            r = make(i, rep)
            if r is not None:
                responses[(name, pid, rep)] = r


def baseline(subject=None, control=None):
    """Two agreeing references plus a subject and a control. Replicate
    jitter (0.002 per replicate) is much smaller than the cross-lane gaps,
    as it is in the field: a subject whose gap to a reference is within
    that reference's own jitter is the same deployment."""
    r = {}
    fill(r, "refA", lambda i, rep: content_shape(["a", "b", "c"], spread=0.002 * rep))
    fill(r, "refB", lambda i, rep: content_shape(["a", "b", "c"], spread=0.04 + 0.002 * rep, chosen=-0.15))
    fill(r, "subj", subject or (lambda i, rep: content_shape(["a", "b", "c"], spread=0.02 + 0.002 * rep)))
    fill(r, "ctrl", control or (lambda i, rep: content_shape(["q", "r"], prompt_tokens=14)), reps=1)
    return r


def http_error(url, code, body=b""):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body))


class StepsTest(unittest.TestCase):
    def test_content_shape(self):
        s = mf.steps(content_shape(["a", "b"], k=3))
        self.assertEqual([t for t, _ in s], ["a", "b"])
        self.assertEqual(len(s[0][1]), 3)
        self.assertAlmostEqual(s[0][1]["a"], -0.1)

    def test_content_shape_without_top_list_degrades_to_chosen(self):
        self.assertEqual(mf.steps(content_shape(["a"], k=1)), [("a", {"a": -0.1})])

    def test_legacy_flat_shape(self):
        d = {"choices": [{"logprobs": {
            "tokens": ["x", "y"], "token_logprobs": [-0.5, -0.7],
            "top_logprobs": [{"x": -0.5, "z": -1.5}, None]}}]}
        self.assertEqual(mf.steps(d), [("x", {"x": -0.5, "z": -1.5}), ("y", {"y": -0.7})])

    def test_no_logprobs(self):
        self.assertEqual(mf.steps(no_logprobs()), [])
        self.assertEqual(mf.topk_width(no_logprobs()), 0)


class ContentSliceTest(unittest.TestCase):
    """Fix 1 (2026-11-03): only the content segment is compared."""

    def toks(self, d):
        s, sliced = mf.content_slice(d)
        return [t for t, _ in s], sliced

    def test_stream_equal_to_content_is_a_noop(self):
        self.assertEqual(self.toks(content_shape(["Par", "is"])), (["Par", "is"], False))

    def test_reasoning_prefix_is_dropped(self):
        d = content_shape(["Paris"], reasoning=["The", " user", " asks", ".", "</think>"])
        self.assertEqual(self.toks(d), (["Paris"], True))

    def test_relay_strips_reasoning_content_but_logprobs_still_cover_it(self):
        d = content_shape(["Paris"], reasoning=["think", "</think>"])
        del d["choices"][0]["message"]["reasoning_content"]
        self.assertEqual(self.toks(d), (["Paris"], True))

    def test_trailing_stop_tokens_are_dropped_without_counting_as_reasoning(self):
        for tail in (["<｜end▁of▁sentence｜>"], ["<|return|>"], ["[e~["],
                     ["<|close|>", "response", "<|sep|>", "<|close|>", "message", "<|sep|>"], [""]):
            d = content_shape(["Paris"], tail=tail)
            self.assertEqual(self.toks(d), (["Paris"], False), tail)

    def test_reasoning_and_stop_tokens_together(self):
        d = content_shape(["Yes", "."], reasoning=["so", " Yes", ".", "</think>"], tail=["<|user|>"])
        self.assertEqual(self.toks(d), (["Yes", "."], True))

    def test_last_occurrence_wins_when_reasoning_repeats_the_answer(self):
        d = content_shape(["Paris"], reasoning=["Paris", " is", " it", "\n\n"])
        s, sliced = mf.content_slice(d)
        self.assertEqual(len(s), 1)
        self.assertTrue(sliced)

    def test_leading_newline_boundary_token_is_dropped(self):
        d = content_shape(["Paris"], reasoning=["think", "</think>", "\n\n"])
        self.assertEqual(self.toks(d), (["Paris"], True))

    def test_partial_boundary_token_is_dropped(self):
        # The token "\n\nPa" straddles the boundary: content starts mid-token.
        d = content_shape(["ris", "."], reasoning=["think", "</think>", "\n\nPa"], content="Paris.")
        self.assertEqual(self.toks(d), (["ris", "."], True))

    def test_empty_content_counts_as_not_aligned_never_distance_one(self):
        d = content_shape([], reasoning=["still", " thinking"], content="")
        s, sliced = mf.content_slice(d)
        self.assertEqual(s, [])
        self.assertTrue(sliced)
        r = mf.compare(d, content_shape(["Paris"]))
        self.assertEqual(r["aligned"], 0)
        self.assertIsNone(r["jsd"])
        self.assertIsNone(r["lp_delta"])

    def test_length_fallback_when_content_is_re_encoded(self):
        # Provider trimmed trailing whitespace: the stream spells "Paris \n"
        # but content is "Paris". The stripped needle still matches.
        d = content_shape(["Paris", " \n"], reasoning=["t", "</think>"], content="Paris")
        self.assertEqual(self.toks(d), (["Paris"], True))
        # Nothing matches verbatim: fall back to the content's length from
        # the end of the stream after the trailing special tokens.
        d = content_shape(["P�", "ris"], reasoning=["t", "</think>"], tail=["<|eos|>"], content="Paris")
        toks, sliced = self.toks(d)
        self.assertTrue(sliced)
        self.assertIn("ris", toks)
        self.assertNotIn("<|eos|>", toks)

    def test_compare_aligns_on_content_across_a_thinking_and_a_non_thinking_lane(self):
        thinking = content_shape(["Paris"], reasoning=["a", "b", "</think>"], tail=["<|eos|>"])
        plain = content_shape(["Paris"])
        r = mf.compare(thinking, plain)
        self.assertEqual(r["aligned"], 1)
        self.assertAlmostEqual(r["jsd"], 0.0, places=9)


class StreamTest(unittest.TestCase):
    """Fix 3: a stream lane aggregates SSE deltas into the non-stream shape."""

    def chunks(self):
        rows = [{"token": "Par", "logprob": -0.1, "top_logprobs": [{"token": "Par", "logprob": -0.1}]},
                {"token": "is", "logprob": -0.2, "top_logprobs": [{"token": "is", "logprob": -0.2}]}]
        return [
            'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hm"},"logprobs":null}]}',
            ': keep-alive',
            'data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "Par"},
                                                "logprobs": {"content": [rows[0]]}}]}),
            'data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "is"},
                                                "logprobs": {"content": [rows[1]]},
                                                "finish_reason": "stop"}]}),
            'data: {"choices":[],"usage":{"prompt_tokens":35,"completion_tokens":2}}',
            'data: [DONE]',
            'data: {"never":"read"}',
        ]

    def test_aggregate(self):
        d = mf.aggregate_stream(mf.parse_sse(self.chunks()))
        self.assertEqual(d["choices"][0]["message"]["content"], "Paris")
        self.assertEqual(d["choices"][0]["message"]["reasoning_content"], "hm")
        self.assertEqual([t for t, _ in mf.steps(d)], ["Par", "is"])
        self.assertEqual(d["usage"]["prompt_tokens"], 35)
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        self.assertEqual(mf.emitted(d), ("hm", "Paris"))

    def test_parse_sse_accepts_bytes(self):
        got = mf.parse_sse([c.encode() for c in self.chunks()])
        self.assertEqual(len(got), 4)

    def test_legacy_flat_logprobs_in_chunks(self):
        chunks = ['data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "x"},
                                                      "logprobs": {"tokens": ["x"], "token_logprobs": [-0.5],
                                                                   "top_logprobs": [{"x": -0.5, "z": -1.5}]}}]}),
                  'data: [DONE]']
        d = mf.aggregate_stream(mf.parse_sse(chunks))
        self.assertEqual(mf.steps(d), [("x", {"x": -0.5, "z": -1.5})])

    def test_stream_lane_sends_stream_options(self):
        body = mf.build_body(lane("l", "reference", stream=True), "hi")
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("stream", mf.build_body(lane("l", "reference"), "hi"))

    def test_post_json_parses_a_stream(self):
        class Resp(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        real = mf.urllib.request.urlopen
        seen = {}

        def fake(req, timeout=None):
            seen["accept"] = req.get_header("Accept")
            return Resp("\n".join(self.chunks()).encode())
        mf.urllib.request.urlopen = fake
        try:
            d = mf.post_json("https://x/v1/chat/completions", "k", {"stream": True, "model": "m"})
        finally:
            mf.urllib.request.urlopen = real
        self.assertEqual(d["choices"][0]["message"]["content"], "Paris")
        self.assertEqual(seen["accept"], "text/event-stream")


class RequestExtraTest(unittest.TestCase):
    """Fix 2: per-model request_extra and max_tokens, per-lane overrides."""

    def test_model_extras_apply_to_every_lane_and_lane_extras_merge_on_top(self):
        entry = {"id": "m", "request_extra": {"reasoning_effort": "none"}, "max_tokens": 2000,
                 "lanes": [lane("a", "subject"), lane("b", "reference", request_extra={"enable_thinking": False}),
                           lane("c", "control", request_extra={"reasoning_effort": None}),
                           dict(lane("d", "reference"), max_tokens=50)]}
        lanes = mf.lanes_of(entry)
        bodies = {l["name"]: mf.build_body(l, "hi") for l in lanes}
        self.assertEqual(bodies["a"]["reasoning_effort"], "none")
        self.assertEqual(bodies["a"]["max_tokens"], 2000)
        self.assertEqual(bodies["b"]["reasoning_effort"], "none")
        self.assertIs(bodies["b"]["enable_thinking"], False)
        self.assertNotIn("reasoning_effort", bodies["c"])  # null removes the key for that lane
        self.assertEqual(bodies["d"]["max_tokens"], 50)

    def test_values_are_literal_pass_through(self):
        l = lane("a", "subject", request_extra={"thinking": {"type": "disabled"}, "n": 1})
        body = mf.build_body(l, "hi")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["n"], 1)

    def test_cache_key_covers_extras(self):
        a = lane("a", "subject", request_extra={"reasoning_effort": "none"})
        b = lane("a", "subject")
        c = dict(lane("a", "subject"), max_tokens=2000)
        d = lane("a", "subject", stream=True)
        paths = {mf.cache_path("d", x, P[0], 0) for x in (a, b, c, d)}
        self.assertEqual(len(paths), 4)

    def test_relay_lane_inherits_model_extras(self):
        l = mf.relay_lane("m", "https://x", request_extra={"reasoning_effort": "none"}, max_tokens=2000)
        body = mf.build_body(l, "hi")
        self.assertEqual((body["reasoning_effort"], body["max_tokens"], body["top_logprobs"]), ("none", 2000, 1))

    def test_run_applies_entry_extras_to_a_relay_lane(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append((url, body))
            return content_shape(["a"], k=1 if "relay" in url else 5)
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "matrix.json"
            f.write_text(json.dumps({"models": [{"id": "m", "request_extra": {"reasoning_effort": "none"},
                                                 "max_tokens": 77, "lanes": [lane("A", "reference"),
                                                                             lane("B", "reference"),
                                                                             lane("C", "control")]}]}))
            mf.run({"model": "m", "lanes_file": str(f), "relay_base": "https://relay"}, env={"K": "k", "DEEPRELAY_API_KEY": "k"},
                   post=post)
        self.assertTrue(bodies)
        self.assertTrue(all(b["reasoning_effort"] == "none" and b["max_tokens"] == 77 for _, b in bodies))

    def test_request_extra_must_be_an_object(self):
        with self.assertRaises(ValueError):
            mf.validate_lanes([lane("a", "subject", request_extra=["x"])])


class JsTest(unittest.TestCase):
    def test_identical_is_zero(self):
        p = {"a": -0.1, "b": -2.5}
        self.assertAlmostEqual(mf.js(p, dict(p)), 0.0, places=9)

    def test_disjoint_is_large_and_bounded(self):
        d = mf.js({"a": 0.0, "z": -8.0}, {"b": 0.0, "y": -8.0})
        self.assertGreater(d, 0.9)
        self.assertLessEqual(d, 1.0 + 1e-9)

    def test_symmetric(self):
        p, q = {"a": -0.2, "b": -1.9}, {"a": -0.6, "c": -1.1}
        self.assertAlmostEqual(mf.js(p, q), mf.js(q, p), places=12)


class CompareTest(unittest.TestCase):
    def test_stops_at_first_divergence(self):
        r = mf.compare(content_shape(["a", "b", "c", "d"]), content_shape(["a", "b", "X", "d"]))
        self.assertEqual(r["aligned"], 2)
        self.assertAlmostEqual(r["prefix_frac"], 0.5)
        self.assertAlmostEqual(r["jsd"], 0.0, places=9)
        self.assertEqual(r["top1_rate"], 1.0)

    def test_no_common_prefix_has_no_distance(self):
        r = mf.compare(content_shape(["a"]), content_shape(["b"]))
        self.assertEqual(r["aligned"], 0)
        self.assertIsNone(r["jsd"])
        self.assertIsNone(r["lp_delta"])
        self.assertIsNone(r["top1_rate"])

    def test_missing_logprobs_has_no_distance(self):
        self.assertIsNone(mf.compare(no_logprobs(), content_shape(["a"]))["jsd"])

    def test_lp_delta(self):
        a, b = content_shape(["a"], k=1), content_shape(["a"], k=1)
        b["choices"][0]["logprobs"]["content"][0]["logprob"] = -0.4
        self.assertAlmostEqual(mf.compare(a, b)["lp_delta"], 0.3)


class WidthTest(unittest.TestCase):
    """Fix 5: distances are width-aware."""

    def test_truncate_keeps_the_most_probable(self):
        self.assertEqual(mf.truncate({"a": -3.0, "b": -0.1, "c": -1.0}, 2), {"b": -0.1, "c": -1.0})
        self.assertEqual(mf.truncate({"a": -3.0}, 5), {"a": -3.0})

    def test_mismatched_widths_compare_at_the_common_k(self):
        wide = content_shape(["a"], k=5)
        narrow = content_shape(["a"], k=2)
        r = mf.compare(wide, narrow)
        self.assertEqual(r["aligned"], 1)
        self.assertAlmostEqual(r["jsd"], 0.0, places=9)  # top-2 of each side agree exactly
        self.assertAlmostEqual(r["jaccard"], 1.0)

    def test_relay_returning_two_entries_when_asked_for_one(self):
        # The relay sometimes answers width 2 to a top_logprobs 1 request;
        # the reference is width 5. The common k is 2, never an error.
        r = mf.compare(content_shape(["a"], k=2), content_shape(["a"], k=5))
        self.assertAlmostEqual(r["jsd"], 0.0, places=9)

    def test_k_cap_applies(self):
        a, b = content_shape(["a"], k=5), content_shape(["a"], k=5, spread=0.5)
        self.assertGreater(mf.compare(a, b)["jsd"], mf.compare(a, b, k=2)["jsd"])
        self.assertIsNone(mf.compare(a, b, k=1)["jsd"])
        self.assertIsNotNone(mf.compare(a, b, k=1)["lp_delta"])

    def test_lane_judged_at_its_own_width_with_a_band_at_that_width(self):
        # References far enough apart that the k=2 band (which only sees
        # the chosen token and one alternative) is not degenerate; the
        # k=2 subject sits between them.
        lanes = LANES + [dict(lane("two", "subject", "S_KEY"), top_logprobs=2)]
        r = {}
        fill(r, "refA", lambda i, rep: content_shape(["a", "b", "c"], spread=0.002 * rep))
        fill(r, "refB", lambda i, rep: content_shape(["a", "b", "c"], spread=0.4 + 0.002 * rep, chosen=-0.3))
        fill(r, "subj", lambda i, rep: content_shape(["a", "b", "c"], spread=0.2 + 0.002 * rep, chosen=-0.2))
        fill(r, "ctrl", lambda i, rep: content_shape(["q", "r"], prompt_tokens=14), reps=1)
        fill(r, "two", lambda i, rep: content_shape(["a", "b", "c"], k=2, spread=0.2 + 0.002 * rep, chosen=-0.2))
        rep = mf.analyze(lanes, r)
        v = rep["lane_verdicts"]["two"]
        self.assertEqual(v["k"], 2)
        self.assertEqual(v["metric"], "jsd")
        self.assertEqual(v["verdict"], "same_model", v)
        self.assertEqual(v["band"], rep["bands_by_k"]["2"]["jsd"])
        self.assertEqual(rep["lane_verdicts"]["subj"]["band"], rep["bands_by_k"]["5"]["jsd"])
        self.assertEqual(set(rep["bands_by_k"]), {"1", "2", "3", "4", "5"})
        self.assertIsNone(rep["bands_by_k"]["1"]["jsd"])
        self.assertIsNotNone(rep["bands_by_k"]["1"]["lp_delta"])
        # The headline band is the first subject's k, as a metric -> float map.
        self.assertEqual(rep["band_k"], 5)
        self.assertEqual(set(rep["band"]), {"jsd", "lp_delta"})

    def test_a_substitute_cannot_hide_at_a_narrow_width(self):
        def far(i, rep):
            d = content_shape(["a", "b", "c"], k=2)
            for row in d["choices"][0]["logprobs"]["content"]:
                row["top_logprobs"] = [{"token": row["token"], "logprob": -0.9},
                                       {"token": "other", "logprob": -1.0}]
            return d
        rep = mf.analyze(LANES, baseline(subject=far))
        self.assertEqual(rep["lane_verdicts"]["subj"]["k"], 2)
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "mismatch")


class SignatureTest(unittest.TestCase):
    """Fix 4: prompt_tokens compared as a relative shape."""

    def test_constant_offset_still_agrees(self):
        # refB prepends a fixed system prompt: +3 on every prompt.
        r = {}
        fill(r, "refA", lambda i, rep: content_shape(["a", "b"], prompt_tokens=40 + i, spread=0.002 * rep))
        fill(r, "refB", lambda i, rep: content_shape(["a", "b"], prompt_tokens=43 + i, spread=0.04 + 0.002 * rep,
                                                     chosen=-0.15))
        fill(r, "subj", lambda i, rep: content_shape(["a", "b"], prompt_tokens=40 + i, spread=0.02 + 0.002 * rep))
        fill(r, "ctrl", lambda i, rep: content_shape(["q"], prompt_tokens=14 + 2 * i), reps=1)
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        v = rep["lane_verdicts"]["subj"]
        self.assertEqual((v["signature_agree"], v["signature_total"]), (12, 12))
        lanes = {l["name"]: l for l in rep["lanes"]}
        self.assertEqual(lanes["refB"]["signature_offset"], 43)
        self.assertEqual(lanes["refB"]["signature"][0], 43)  # absolute counts are kept
        self.assertTrue(rep["controls"][0]["by_signature"])

    def test_a_different_tokenizer_still_mismatches(self):
        rep = mf.analyze(LANES, baseline(
            subject=lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=40 + (i % 3))))
        self.assertEqual(rep["verdict"], "mismatch")
        self.assertEqual(rep["lane_verdicts"]["subj"]["evidence"], "signature")

    def test_subject_without_the_anchor_prompt_is_re_anchored(self):
        r = baseline()
        r.pop(("subj", P[0], 0)); r.pop(("subj", P[0], 1))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["lane_verdicts"]["subj"]["signature_total"], 11)

    def test_control_with_the_same_shape_does_not_separate_by_signature(self):
        rep = mf.analyze(LANES, baseline(
            control=lambda i, rep: content_shape(["q", "r"], prompt_tokens=99)))
        self.assertFalse(rep["controls"][0]["by_signature"])

    def test_control_separates_by_signature_on_two_prompts_not_one(self):
        # Two English-efficient tokenizers agree on most English prompts and
        # differ on the CJK / Cyrillic / stress prompts (measured 4/12 for
        # gpt-oss vs DeepSeek). One flaky prompt is not separation; two are.
        def ctrl(diffs):
            return lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=99 + (1 if i in diffs else 0),
                                                spread=0.015)
        one = mf.analyze(LANES, baseline(control=ctrl({3})))
        two = mf.analyze(LANES, baseline(control=ctrl({3, 10})))
        self.assertFalse(one["controls"][0]["by_signature"])
        self.assertTrue(two["controls"][0]["by_signature"])
        self.assertFalse(one["controls"][0]["separates"])
        self.assertTrue(two["controls"][0]["separates"])
        self.assertEqual(one["verdict"], "inconclusive")
        # Separating by signature alone still leaves jsd undemonstrated.
        self.assertEqual(two["verdict"], "inconclusive")
        self.assertIn("no control separates on jsd", two["reasons"][0])
        self.assertAlmostEqual(mf.CONTROL_SIGNATURE_SEPARATION * len(mf.PROMPTS), 2.0)

    def test_a_reference_that_is_the_subjects_own_upstream_is_not_independent(self):
        # Measured 2026-11-03: deeprelay/glm-5.2 vs a candidate reference
        # that was the relay's own upstream, identical content 12/12 at
        # |dlp| 7e-7. The band it defines is "relay ==
        # upstream", which any build satisfies.
        r = baseline()
        for pid in P:
            for rep in range(2):
                r[("refB", pid, rep)] = json.loads(json.dumps(r[("subj", pid, rep)]))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["verdict"], "insufficient")
        self.assertEqual(rep["dependent_references"][0]["reference"], "refB")
        self.assertTrue(any("not independent" in x for x in rep["reasons"]), rep["reasons"])
        # Three different answers and a real distance: independent.
        r = baseline(subject=lambda i, rep: content_shape(["a", "b", "c"] if i > 2 else ["z"],
                                                          spread=0.02 + 0.01 * rep))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["dependent_references"], [])

    def test_dependence_is_relative_to_the_references_own_noise(self):
        # Measured 2026-11-03, run 3: deeprelay/glm-5.2 vs its own upstream
        # at |dlp| 0.0016 with that deployment's replicate noise 0.0016. A relay
        # to the same deployment measures that deployment's jitter, not 0.
        r = baseline()
        fill(r, "refB", lambda i, rep: content_shape(["a", "b", "c"], spread=0.02 + 0.004 * rep, chosen=-0.15))
        fill(r, "subj", lambda i, rep: content_shape(["a", "b", "c"], spread=0.022 + 0.004 * rep, chosen=-0.15))
        rep = mf.analyze(LANES, r)
        self.assertEqual([d["reference"] for d in rep["dependent_references"]], ["refB"])
        self.assertEqual(rep["verdict"], "insufficient")
        # Same answers but a gap larger than the jitter: an independent deployment.
        fill(r, "subj", lambda i, rep: content_shape(["a", "b", "c"], spread=0.05 + 0.004 * rep, chosen=-0.15))
        self.assertEqual(mf.analyze(LANES, r)["dependent_references"], [])
        # Measured 2026-11-03, run 4: a NOISY reference (Fireworks jitter
        # 0.017) must not widen "same deployment" until an independent
        # subject at 0.023 qualifies. The tolerance is capped.
        fill(r, "refB", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.15 - 0.03 * rep))
        fill(r, "subj", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.17))
        fill(r, "refA", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.10 - 0.001 * rep))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["dependent_references"], [], rep.get("reasons"))


class AdvisoryControlTest(unittest.TestCase):
    """A sibling checkpoint at the edge of the metric's resolution is
    measured every run but does not gate the verdict."""

    def test_advisory_control_is_reported_but_does_not_gate(self):
        lanes = LANES + [lane("sib", "control", "A_KEY", advisory=True)]
        r = baseline()
        fill(r, "sib", lambda i, rep: content_shape(["a", "b", "c"], spread=0.015), reps=1)
        rep = mf.analyze(lanes, r)
        sib = next(c for c in rep["controls"] if c["lane"] == "sib")
        self.assertTrue(sib["advisory"])
        self.assertFalse(sib["separates"])
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertTrue(any("advisory control sib did not separate" in n for n in rep["notes"]))
        self.assertIn("(advisory)", mf.render(rep, r))

    def test_advisory_controls_alone_do_not_demonstrate_separation(self):
        lanes = [dict(l, advisory=True) if l["role"] == "control" else l for l in LANES]
        rep = mf.analyze(lanes, baseline())
        self.assertEqual(rep["verdict"], "inconclusive")
        self.assertTrue(any("no gating negative control" in x for x in rep["reasons"]))

    def test_gating_control_still_gates(self):
        rep = mf.analyze(LANES, baseline(
            control=lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=40, spread=0.015)))
        self.assertEqual(rep["verdict"], "inconclusive")


class VerdictTest(unittest.TestCase):
    def test_same_model_passes(self):
        rep = mf.analyze(LANES, baseline())
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "same_model")
        self.assertEqual(rep["lane_verdicts"]["subj"]["evidence"], "distribution")
        self.assertTrue(rep["controls"][0]["separates"])
        self.assertEqual(rep["harness_version"], mf.HARNESS_VERSION)

    def test_signature_mismatch_fails(self):
        rep = mf.analyze(LANES, baseline(
            subject=lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=14 + i)))
        self.assertEqual(rep["verdict"], "mismatch")
        self.assertEqual(rep["lane_verdicts"]["subj"]["evidence"], "signature")

    def test_top1_over_the_limit_is_inconclusive_never_mismatch(self):
        # GitHub run 41724166806 (2026-11-03): subject |dlp| 0.0184/0.0161,
        # band 0.0062, self-noise 0.0115, limit 0.0173, mean 0.01725: a hair
        # over on noise. Width 1 may confirm a match, never fail a lane.
        lanes = LANES + [dict(lane("one", "subject", "S_KEY"), top_logprobs=1)]
        r = baseline()
        fill(r, "one", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.9))
        rep = mf.analyze(lanes, r)
        v = rep["lane_verdicts"]["one"]
        self.assertEqual((v["metric"], v["k"]), ("lp_delta", 1))
        self.assertGreater(v["distance"], v["limit"])
        self.assertEqual(v["verdict"], "inconclusive")
        self.assertIn("wider top_logprobs", v["reason"])
        self.assertEqual(rep["verdict"], "inconclusive")
        self.assertNotIn("mismatch", {x["verdict"] for x in rep["lane_verdicts"].values()})
        # the same lane at width 1 inside the limit still confirms a match
        fill(r, "one", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.11))
        self.assertEqual(mf.analyze(lanes, r)["lane_verdicts"]["one"]["verdict"], "same_model")
        # and a signature difference still fails a width-1 lane outright
        fill(r, "one", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.11, prompt_tokens=40 + i))
        self.assertEqual(mf.analyze(lanes, r)["lane_verdicts"]["one"]["verdict"], "mismatch")

    def test_distribution_mismatch_fails(self):
        def far(i, rep):
            d = content_shape(["a", "b", "c"])
            for row in d["choices"][0]["logprobs"]["content"]:
                row["top_logprobs"] = [{"token": row["token"], "logprob": -1.2}] + [
                    {"token": f"other{j}", "logprob": -1.3} for j in range(4)]
            return d
        rep = mf.analyze(LANES, baseline(subject=far))
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "mismatch")
        self.assertEqual(rep["verdict"], "mismatch")

    def test_control_that_does_not_separate_is_inconclusive(self):
        rep = mf.analyze(LANES, baseline(
            control=lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=40, spread=0.015)))
        self.assertFalse(rep["controls"][0]["separates"])
        self.assertEqual(rep["verdict"], "inconclusive")

    def test_no_control_is_inconclusive(self):
        lanes = [l for l in LANES if l["role"] != "control"]
        r = {k: v for k, v in baseline().items() if k[0] != "ctrl"}
        self.assertEqual(mf.analyze(lanes, r)["verdict"], "inconclusive")

    def test_relay_top1_uses_chosen_token_logprob(self):
        lanes = LANES + [mf.relay_lane("m", "https://api.example/v1")]
        r = baseline()
        fill(r, "relay", lambda i, rep: content_shape(["a", "b", "c"], k=1, chosen=-0.11))
        rep = mf.analyze(lanes, r)
        self.assertEqual(rep["lane_verdicts"]["relay"]["evidence"], "top1_logprob")
        self.assertEqual(rep["lane_verdicts"]["relay"]["verdict"], "same_model")
        self.assertEqual(rep["lane_verdicts"]["relay"]["k"], 1)

    def test_one_reference_is_insufficient(self):
        r = {k: v for k, v in baseline().items() if k[0] != "refB"}
        self.assertEqual(mf.analyze(LANES, r)["verdict"], "insufficient")

    def test_references_without_logprobs_are_insufficient(self):
        r = baseline()
        fill(r, "refB", lambda i, rep: no_logprobs())
        self.assertEqual(mf.analyze(LANES, r)["verdict"], "insufficient")

    def test_references_with_disjoint_prompts_do_not_crash(self):
        r = baseline()
        for pid in P[6:]:
            r.pop(("refA", pid, 0)); r.pop(("refA", pid, 1))
        for pid in P[:6]:
            r.pop(("refB", pid, 0)); r.pop(("refB", pid, 1))
        self.assertEqual(mf.analyze(LANES, r)["verdict"], "insufficient")

    def test_reasoning_exhausting_max_tokens_is_inconclusive_with_the_count(self):
        r = baseline(subject=lambda i, rep: content_shape([], reasoning=["still", " thinking"], content=""))
        rep = mf.analyze(LANES, r)
        v = rep["lane_verdicts"]["subj"]
        self.assertEqual(v["verdict"], "inconclusive")
        self.assertEqual(v["empty_content_prompts"], 12)
        self.assertIn("empty content", v["reason"])
        self.assertNotEqual(rep["verdict"], "pass")

    def test_thinking_on_lane_cannot_be_failed_on_distance_alone(self):
        # Measured 2026-11-03 on GLM 5.3: with thinking on, each lane's
        # content conditions on its own reasoning, so a distance over the
        # limit is inconclusive, never a mismatch. The signature still fails it.
        def far(i, rep, think=True, pt=40):
            d = content_shape(["a", "b", "c"], reasoning=["t", "</think>"] if think else None, prompt_tokens=pt)
            for row in d["choices"][0]["logprobs"]["content"][-3:]:
                row["top_logprobs"] = [{"token": row["token"], "logprob": -1.2}] + [
                    {"token": f"other{j}", "logprob": -1.3} for j in range(4)]
            return d
        rep = mf.analyze(LANES, baseline(subject=far))
        v = rep["lane_verdicts"]["subj"]
        self.assertEqual(v["verdict"], "inconclusive")
        self.assertTrue(v["thinking"])
        self.assertEqual(v["thinking_prompts"], 12)
        self.assertIn("thinking was on", v["reason"])
        self.assertEqual(rep["verdict"], "inconclusive")
        # reasoning_content beside content-only logprobs counts as thinking too
        def far_rc(i, rep):
            d = far(i, rep, think=False)
            d["choices"][0]["message"]["reasoning_content"] = "hmm"
            return d
        self.assertEqual(mf.analyze(LANES, baseline(subject=far_rc))["lane_verdicts"]["subj"]["verdict"],
                         "inconclusive")
        # without thinking the same distance is a mismatch
        self.assertEqual(mf.analyze(LANES, baseline(subject=lambda i, rep: far(i, rep, think=False)))
                         ["lane_verdicts"]["subj"]["verdict"], "mismatch")
        # and a signature difference still fails a thinking lane
        self.assertEqual(mf.analyze(LANES, baseline(subject=lambda i, rep: far(i, rep, pt=40 + i)))
                         ["lane_verdicts"]["subj"]["verdict"], "mismatch")

    def test_reasoning_prompts_are_counted_per_lane(self):
        r = baseline(subject=lambda i, rep: content_shape(["a", "b", "c"], reasoning=["t", "</think>"],
                                                          spread=0.02 + 0.002 * rep))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        lanes = {l["name"]: l for l in rep["lanes"]}
        self.assertEqual(lanes["subj"]["reasoning_prompts"], 12)
        self.assertEqual(lanes["refA"]["reasoning_prompts"], 0)

    def test_skipped_is_a_list_of_names_with_details_beside_it(self):
        rep = mf.analyze(LANES, baseline(), skipped=[{"lane": "x", "role": "control", "reason": "X is not set"}])
        self.assertEqual(rep["skipped"], ["x"])
        self.assertEqual(rep["skipped_lanes"][0]["reason"], "X is not set")
        self.assertIn("noise_cap_x", rep["thresholds"])


class ReviewerCasesTest(unittest.TestCase):
    """Synthetic adversarial cases from the original review. Each must NOT
    pass."""

    SPEC = [lane("A", "reference"), lane("B", "reference"),
            lane("S", "subject"), lane("C", "control")]

    def build(self, f, lanes=None):
        out = {}
        for l in lanes or self.SPEC:
            fill(out, l["name"], lambda i, rep, n=l["name"]: f(n, i, rep))
        return out

    def test_1_subject_identical_to_control_cannot_pass(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            if n == "C":
                return dist_shape(FAR) if rep == 0 else None
            if rep == 1 and i < 4:
                return dist_shape(FAR, first="y")  # noisy subject
            return dist_shape(FAR)
        rep = mf.analyze(self.SPEC, self.build(f))
        self.assertNotEqual(rep["verdict"], "pass")
        self.assertNotEqual(rep["lane_verdicts"]["S"]["verdict"], "same_model")

    def test_1_identical_subject_and_control_with_a_noise_inflated_limit(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            mid = {"x": .80, "y": .20}
            if n == "C":
                return dist_shape(mid) if rep == 0 else None
            return dist_shape(mid if rep == 0 else {"x": .70, "y": .30})
        rep = mf.analyze(self.SPEC, self.build(f))
        self.assertNotEqual(rep["verdict"], "pass", rep["reasons"])

    def test_2_drifted_reference_band_over_ceiling_is_inconclusive(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(FAR)
            if n == "C":
                return dist_shape({"x": .99, "y": .01}, sig=99 + i) if rep == 0 else None
            return dist_shape({"x": .5, "y": .5})
        rep = mf.analyze(self.SPEC, self.build(f))
        self.assertGreater(rep["band"]["jsd"], mf.JS_BAND_CEILING)
        self.assertEqual(rep["verdict"], "inconclusive")
        self.assertNotEqual(rep["lane_verdicts"]["S"]["verdict"], "same_model")

    def test_2_first_token_divergence_does_not_loosen_the_floor(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            if n == "C":
                return dist_shape(REF_A, sig=99 + i) if rep == 0 else None
            if rep == 1 and i < 8:
                return dist_shape(FAR, first="y")
            return dist_shape(FAR)
        rep = mf.analyze(self.SPEC, self.build(f))
        v = rep["lane_verdicts"]["S"]
        self.assertLess(v["limit"], mf.JS_LIMIT_CEILING)
        self.assertEqual(v["verdict"], "mismatch")
        self.assertEqual(rep["noise"]["S"]["aligned"], 4)

    def test_3_subject_without_logprobs_is_inconclusive(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            if n == "C":
                return dist_shape(REF_A, sig=99 + i) if rep == 0 else None
            return dist_shape({"x": .5, "y": .5}, logprobs=False)
        rep = mf.analyze(self.SPEC, self.build(f))
        self.assertEqual(rep["lane_verdicts"]["S"]["evidence"], "signature_only")
        self.assertEqual(rep["lane_verdicts"]["S"]["verdict"], "inconclusive")
        self.assertEqual(rep["verdict"], "inconclusive")

    def test_3_relay_fallback_caps_below_pass(self):
        lanes = LANES + [mf.relay_lane("m", "https://api.example")]
        r = baseline()
        fill(r, "relay", lambda i, rep: dict(no_logprobs(), _fingerprint_logprobs_requested=False))
        rep = mf.analyze(lanes, r)
        self.assertEqual(rep["lane_verdicts"]["relay"]["verdict"], "inconclusive")
        self.assertIn("fallback", rep["lane_verdicts"]["relay"]["reason"])
        self.assertEqual(rep["verdict"], "inconclusive")

    def test_4_subject_with_one_prompt_is_insufficient(self):
        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            if n == "C":
                return dist_shape(REF_A, sig=99 + i) if rep == 0 else None
            return dist_shape(REF_A) if i == 0 else None
        self.assertEqual(mf.analyze(self.SPEC, self.build(f))["verdict"], "insufficient")

    def test_4_errored_subject_does_not_pass_on_the_relay_alone(self):
        lanes = self.SPEC + [dict(lane("relay", "relay"), top_logprobs=1)]

        def f(n, i, rep):
            if n == "A":
                return dist_shape(REF_A)
            if n == "B":
                return dist_shape(REF_B)
            if n == "C":
                return dist_shape(REF_A, sig=99 + i) if rep == 0 else None
            if n == "S":
                return None  # e.g. a revoked key: 24 auth_errors
            return dist_shape(SUBJ)
        rep = mf.analyze(lanes, self.build(f, lanes))
        self.assertEqual(rep["verdict"], "insufficient")
        self.assertTrue(any("subject lane S" in x for x in rep["reasons"]))

    def test_4_skipped_subject_is_insufficient(self):
        def post(url, key, body, **_):
            return content_shape(["a"], prompt_tokens=99 if "ctrl" in url else 40)
        with tempfile.TemporaryDirectory() as d:
            rep = mf.run({"model": "m", "raw_dir": d}, lanes=LANES,
                         env={"A_KEY": "k", "B_KEY": "k"}, post=post)
        self.assertEqual(rep["skipped"], ["subj"])
        self.assertEqual(rep["verdict"], "insufficient")


def two_post(model_lp):
    """post() answering per model from {model: (distribution, signature)}.
    The control's signature varies per prompt so its shape differs."""
    def post(url, key, body, **_):
        dist_, sig = model_lp[body["model"]]
        if sig == 99:
            sig += P.index(next(p for p in P if mf.PROMPTS[p] == body["messages"][0]["content"]))
        return dist_shape(dist_, sig=sig)
    return post


# The subject sits a hair off refA: two deployments of the same weights
# measure 0.006-0.05 apart, never 0 (0 means the same deployment).
SUBJ = {"x": .895, "y": .105}
GOOD = {"refA": (REF_A, 10), "refB": (REF_B, 10), "subj": (SUBJ, 10),
        "ctrl": ({"z": .9, "w": .1}, 99)}


class RunTest(unittest.TestCase):
    def test_missing_keys_are_skipped_and_never_pass(self):
        def post(url, key, body, **_):
            raise AssertionError("no lane has a key; nothing may be sent")
        rep = mf.run({"model": "m"}, lanes=LANES, env={}, post=post)
        self.assertEqual(set(rep["skipped"]), {"subj", "refA", "refB", "ctrl"})
        self.assertEqual(rep["verdict"], "insufficient")

    def test_5_normal_runs_never_reuse_saved_responses(self):
        calls = []

        def good(url, key, body, **_):
            calls.append(url)
            return two_post(GOOD)(url, key, body)

        def bad(url, key, body, **_):
            calls.append(url)
            return two_post(dict(GOOD, subj=(FAR, 10)))(url, key, body)
        env = {"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}
        with tempfile.TemporaryDirectory() as d:
            r1 = mf.run({"model": "m", "raw_dir": d}, lanes=LANES, env=env, post=good)
            first = len(calls)
            r2 = mf.run({"model": "m", "raw_dir": d}, lanes=LANES, env=env, post=bad)
            again = mf.run({"model": "m", "raw_dir": d, "analyze_only": True}, lanes=LANES, env=env,
                           post=lambda *a, **k: self.fail("analyze_only must not send"))
        self.assertEqual(r1["verdict"], "pass", r1["reasons"])
        self.assertEqual(len(calls) - first, first)  # the second run re-sent everything
        self.assertEqual(r2["verdict"], "mismatch")
        self.assertEqual(again["verdict"], "mismatch")  # reads the latest saved responses
        self.assertEqual(again["source"]["mode"], "analyze_only")
        self.assertEqual(again["source"]["responses_loaded"], first)
        self.assertIn("nothing was sent", again["text"])

    def test_5_cache_key_includes_the_logprobs_mode(self):
        l = mf.relay_lane("m", "https://x")
        self.assertNotEqual(mf.cache_path("d", l, "p01_capital", 0, True),
                            mf.cache_path("d", l, "p01_capital", 0, False))

    def test_6_fallback_calls_count_against_max_requests(self):
        calls = []

        def post(url, key, body, **_):
            calls.append(body)
            if body.get("logprobs"):
                raise http_error(url, 400, json.dumps({"error": {
                    "message": "deeprelay/m does not support logprobs.",
                    "type": "invalid_request_error", "param": "logprobs"}}).encode())
            return dist_shape(REF_A)
        rep = mf.run({"model": "m", "max_requests": 24},
                     lanes=[mf.relay_lane("m", "https://relay.example")],
                     env={"DEEPRELAY_API_KEY": "k"}, post=post)
        self.assertLessEqual(len(calls), 24)
        self.assertEqual(rep["requests"]["sent"], len(calls))
        self.assertTrue(any("max_requests" in (e["taxonomy"], e.get("retry_skipped"))
                            for e in rep["requests"]["errors"]))

    def test_planned_requests_over_the_ceiling_is_an_error_not_a_raise(self):
        def post(url, key, body, **_):
            raise AssertionError("must not send past the ceiling")
        rep = mf.run({"model": "m", "max_requests": 5}, lanes=LANES,
                     env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}, post=post)
        self.assertEqual(rep["verdict"], "error")
        self.assertIn("max-requests", rep["reasons"][0])

    def test_7_deadline_turns_unfinished_probes_into_errors(self):
        def post(url, key, body, timeout=None, **_):
            time.sleep(0.3)
            return content_shape(["a"])
        t0 = time.monotonic()
        rep = mf.run({"model": "m", "deadline_s": 0.5, "workers": 2}, lanes=LANES,
                     env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}, post=post)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertTrue(any(e["taxonomy"] == "deadline" for e in rep["requests"]["errors"]))
        self.assertEqual(rep["verdict"], "insufficient")
        self.assertIn("VERDICT", rep["text"])

    def test_8_run_never_raises(self):
        rep = mf.run({"model": "no/such-model"}, env={})
        self.assertEqual(rep["verdict"], "error")
        self.assertIn("KeyError", rep["reasons"][0])
        self.assertEqual(rep["harness_version"], mf.HARNESS_VERSION)

    def test_8_unreadable_saved_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            l = LANES[1]
            path = mf.cache_path(pathlib.Path(d) / "m" / "20261014T000000000000000Z", l, P[0], 0, True)
            path.parent.mkdir(parents=True)
            path.write_text("{not json")
            rep = mf.run({"model": "m", "raw_dir": d, "analyze_only": True}, lanes=LANES, env={})
        self.assertEqual(rep["verdict"], "insufficient")


class ReReviewTest(unittest.TestCase):
    def test_I1_no_verdict_lane_never_passes(self):
        lanes = [lane("A", "reference"), lane("B", "reference"), lane("C", "control")]
        r = {}
        fill(r, "A", lambda i, rep: dist_shape(REF_A))
        fill(r, "B", lambda i, rep: dist_shape(REF_B))
        fill(r, "C", lambda i, rep: dist_shape(REF_A, sig=99 + i), reps=1)
        rep = mf.analyze(lanes, r)
        self.assertEqual(rep["verdict"], "insufficient")
        self.assertIn("no subject lane", rep["reasons"][0])
        via_run = mf.run({}, lanes=lanes, env={"K": "k"}, post=two_post(
            {"A": (REF_A, 10), "B": (REF_B, 10), "C": (REF_A, 99)}))
        self.assertEqual(via_run["verdict"], "insufficient")

    def test_I2_analyze_only_reads_only_the_latest_run(self):
        lanes = LANES
        env = {"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}
        good = two_post(GOOD)

        def bad(url, key, body, **kw):
            if body["model"] == "subj":
                raise http_error(url, 401)
            return good(url, key, body, **kw)
        with tempfile.TemporaryDirectory() as d:
            r1 = mf.run({"raw_dir": d, "model": "m"}, lanes=lanes, env=env, post=good)
            r2 = mf.run({"raw_dir": d, "model": "m"}, lanes=lanes, env=env, post=bad)
            latest = mf.run({"raw_dir": d, "model": "m", "analyze_only": True}, lanes=lanes, env={})
            named = mf.run({"raw_dir": d, "model": "m", "analyze_only": True,
                            "run": r1["source"]["run"]}, lanes=lanes, env={})
        self.assertEqual(r1["verdict"], "pass", r1["reasons"])
        self.assertEqual(r2["verdict"], "insufficient")
        self.assertNotEqual(r1["source"]["run"], r2["source"]["run"])
        self.assertEqual(latest["verdict"], "insufficient")
        self.assertEqual(latest["source"]["run"], r2["source"]["run"])
        self.assertIn(r2["source"]["run"], latest["text"].splitlines()[0])
        self.assertEqual(named["verdict"], "pass")

    def test_I2_newer_fallback_file_beats_older_logprobs_file(self):
        l = mf.relay_lane("m", "https://x")
        with tempfile.TemporaryDirectory() as d:
            mf.write_atomic(mf.cache_path(d, l, P[0], 0, True), content_shape(["a"]))
            time.sleep(0.01)
            mf.write_atomic(mf.cache_path(d, l, P[0], 0, False), no_logprobs())
            got = mf.load_saved([l], pathlib.Path(d), 1)
        self.assertEqual(mf.topk_width(got[("relay", P[0], 0)]), 0)

    def test_I2_analyze_only_without_a_saved_run_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            rep = mf.run({"raw_dir": d, "model": "m", "analyze_only": True}, lanes=LANES, env={})
        self.assertEqual(rep["verdict"], "error")

    def test_N1_own_noise_cannot_buy_a_pass(self):
        def q_for(target, base=.9):
            lo, hi = 0.5, base
            for _ in range(60):
                mid = (lo + hi) / 2
                if mf.js({"x": math.log(base), "y": math.log(1 - base)},
                         {"x": math.log(mid), "y": math.log(1 - mid)}) > target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        lanes = [lane("A", "reference"), lane("B", "reference"),
                 lane("S", "subject"), lane("C", "control")]
        for d_target, noise_target in [(0.18, 0.13), (0.12, 0.10), (0.20, 0.14)]:
            qS = q_for(d_target)
            qN = q_for(noise_target, qS)
            r = {}
            fill(r, "A", lambda i, rep: dist_shape({"x": .90, "y": .10}))
            fill(r, "B", lambda i, rep: dist_shape({"x": .88, "y": .12}))
            fill(r, "C", lambda i, rep: dist_shape({"z": .9, "w": .1}, sig=99 + i), reps=1)
            fill(r, "S", lambda i, rep, a=qS, b=qN: dist_shape(
                {"x": a, "y": 1 - a} if rep == 0 else {"x": b, "y": 1 - b}))
            rep = mf.analyze(lanes, r)
            # A noisy independent subject is never read as "same deployment".
            self.assertEqual(rep["dependent_references"], [], (d_target, noise_target))
            v = rep["lane_verdicts"]["S"]
            self.assertNotEqual(v["verdict"], "same_model", (d_target, v))
            self.assertNotEqual(rep["verdict"], "pass")
            self.assertLessEqual(v["limit"], mf.JS_TOLERANCE * mf.NOISE_CAP_X * rep["band"]["jsd"] + 1e-12)

    def test_N2_transient_failures_are_retried(self):
        seen = {}

        def flaky(url, key, body, **kw):
            k = (body["model"], body["messages"][0]["content"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= 2 and body["model"] == "subj":
                raise http_error(url, 429)
            return two_post(GOOD)(url, key, body, **kw)
        rep = mf.run({"model": "m"}, lanes=LANES,
                     env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}, post=flaky)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["requests"]["errors"], [])

    def test_N2_a_failure_past_MAX_RETRIES_is_final_for_the_probe(self):
        calls, slept = [], []

        def post(url, key, body, **_):
            calls.append(1)
            raise http_error(url, 429)
        real_sleep = mf.time.sleep
        mf.time.sleep = slept.append
        try:
            _, rec = mf.call_lane(LANES[1], "k", P[0], 0, None, post=post)
        finally:
            mf.time.sleep = real_sleep
        self.assertEqual((rec["status"], rec["taxonomy"], len(calls), rec["retries"]),
                         ("error", "rate_limited", 1 + mf.MAX_RETRIES, mf.MAX_RETRIES))
        self.assertEqual(len(slept), mf.MAX_RETRIES)
        self.assertEqual(slept, sorted(slept))  # backoff grows

    def test_N2_a_probe_that_failed_every_retry_is_sent_again_after_the_sweep(self):
        seen = {}

        def overloaded_for_a_while(url, key, body, **kw):
            k = (body["model"], body["messages"][0]["content"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= 1 + mf.MAX_RETRIES and body["model"] == "subj":
                raise http_error(url, 503)
            return two_post(GOOD)(url, key, body, **kw)
        lines = []
        # One worker so the first replicate of each subject prompt burns every
        # retry before the second replicate even starts.
        rep = mf.run({"model": "m", "workers": 1}, lanes=LANES,
                     env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"},
                     post=overloaded_for_a_while, log=lines.append)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["requests"]["errors"], [])
        self.assertEqual(rep["source"]["reprobed"], len(P))
        again = [l for l in lines if l.startswith("REPROBE:")]
        self.assertEqual(len(again), 1, lines)
        self.assertIn(f"{len(P)} probes on subj", again[0])
        self.assertTrue(any("re-probe" in l and "plan:" in l for l in lines), lines)
        self.assertIn(f"{len(P)} re-probed after the sweep", mf.render(rep, {}))

    def test_N2_the_re_probe_pass_replaces_the_failed_record_and_sums_its_calls(self):
        n = []

        def post(url, key, body, **_):
            n.append(1)
            if len(n) <= 1 + mf.MAX_RETRIES:
                raise http_error(url, 503)
            return content_shape(["a"])
        responses, records, skipped = mf.collect(
            [LANES[1]], {"A_KEY": "k"}, None, 1, 100, workers=1, post=post)
        rec = [r for r in records if r["prompt"] == P[0]][0]
        self.assertEqual((rec["status"], rec["reprobed"], rec["first_error"]),
                         ("ok", True, "http_5xx"))
        self.assertEqual(rec["http_calls"], 2 + mf.MAX_RETRIES)
        self.assertEqual(rec["retries"], mf.MAX_RETRIES)
        self.assertEqual(len(records), len(P))  # replaced, not appended
        self.assertEqual(len(responses), len(P))

    def test_N2_the_re_probe_pass_leaves_non_retryable_errors_alone(self):
        lines, n = [], []

        def post(url, key, body, **_):
            n.append(1)
            if body["messages"][0]["content"] == mf.PROMPTS[P[0]]:
                raise http_error(url, 401)
            return content_shape(["a"])
        responses, records, _ = mf.collect(
            [LANES[1]], {"A_KEY": "k"}, None, 1, 100, workers=1, post=post, log=lines.append)
        self.assertEqual(len(n), len(P))
        self.assertFalse(any(l.startswith("REPROBE") for l in lines), lines)
        bad = [r for r in records if r["status"] == "error"]
        self.assertEqual([(r["taxonomy"], r.get("reprobed")) for r in bad], [("auth_error", None)])

    def test_N2_the_re_probe_pass_is_skipped_when_the_budget_is_spent(self):
        lines = []

        def post(url, key, body, **_):
            raise http_error(url, 503)
        _, records, _ = mf.collect(
            [LANES[1]], {"A_KEY": "k"}, None, 1, len(P), workers=1, post=post, log=lines.append)
        self.assertTrue(any(l.startswith("REPROBE skipped") and "max-requests" in l
                            for l in lines), lines)
        self.assertEqual(sum(1 for r in records if r["status"] == "error"), len(P))
        self.assertFalse(any(r.get("reprobed") for r in records))

    def test_N2_no_retry_on_auth_errors(self):
        calls = []

        def post(url, key, body, **_):
            calls.append(1)
            raise http_error(url, 401)
        _, rec = mf.call_lane(LANES[1], "k", P[0], 0, None, post=post)
        self.assertEqual((rec["taxonomy"], len(calls)), ("auth_error", 1))

    def test_N2_coverage_counts_any_replicate(self):
        r = baseline()
        for pid in P[:4]:  # r0 lost on 4 prompts; r1 survived
            r.pop(("subj", pid, 0))
        rep = mf.analyze(LANES, r)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["lane_verdicts"]["subj"]["prompts"], 12)

    def test_N3_dropped_control_is_noted(self):
        lanes = LANES + [lane("ctrl2", "control", "A_KEY")]
        r = baseline()
        fill(r, "ctrl2", lambda i, rep: content_shape(["q"], prompt_tokens=77) if i < 3 else None, reps=1)
        rep = mf.analyze(lanes, r)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertTrue(any("control ctrl2 dropped" in n for n in rep["notes"]))
        self.assertIn("note: control ctrl2 dropped", mf.render(rep, r))

    def test_N4_deadline_sweep_keeps_probes_that_finished_late(self):
        real = mf.concurrent.futures.as_completed

        def give_up(fs, timeout=None):
            mf.concurrent.futures.wait(list(fs))
            raise mf.concurrent.futures.TimeoutError()
        mf.concurrent.futures.as_completed = give_up
        try:
            rep = mf.run({"model": "m"}, lanes=LANES,
                         env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}, post=two_post(GOOD))
        finally:
            mf.concurrent.futures.as_completed = real
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(rep["requests"]["errors"], [])

    def test_N5_band_ceiling_is_the_binding_reason(self):
        self.assertAlmostEqual(mf.JS_BAND_CEILING * mf.JS_TOLERANCE, mf.JS_LIMIT_CEILING, places=3)
        lanes = [lane("A", "reference"), lane("B", "reference"),
                 lane("S", "subject"), lane("C", "control")]
        r = {}
        fill(r, "A", lambda i, rep: dist_shape({"x": .90, "y": .10}))
        fill(r, "B", lambda i, rep: dist_shape({"x": .76, "y": .24}))
        fill(r, "S", lambda i, rep: dist_shape({"x": .895, "y": .105}))
        fill(r, "C", lambda i, rep: dist_shape({"z": .9, "w": .1}, sig=99 + i), reps=1)
        rep = mf.analyze(lanes, r)
        self.assertGreater(rep["band"]["jsd"], mf.JS_BAND_CEILING)
        self.assertEqual(rep["verdict"], "inconclusive")
        self.assertIn("reference band", rep["lane_verdicts"]["S"]["reason"])


class ThirdReviewTest(unittest.TestCase):
    def test_unknown_role_is_rejected_not_skipped(self):
        lanes = [lane("A", "reference"), lane("B", "reference"),
                 lane("S", "subjects"), lane("C", "control"),
                 dict(mf.relay_lane("R", "http://x"), key_env="K")]

        def post(url, key, body, **_):
            if body["model"] == "C":
                return dist_shape({"z": .9, "w": .1}, sig=99)
            if body["model"] == "S":
                raise http_error(url, 401)
            return dist_shape({"A": REF_A, "B": REF_B}.get(body["model"], REF_A))
        rep = mf.run({}, lanes=lanes, env={"K": "k"}, post=post)
        self.assertEqual(rep["verdict"], "error")
        self.assertIn("unknown role 'subjects'", rep["reasons"][0])

    def test_load_lanes_validates_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "lanes.json"
            f.write_text(json.dumps({"models": {"m": {"lanes": [
                dict(lane("A", "reference")), dict(lane("S", "subjects"))]}}}))
            with self.assertRaises(ValueError):
                mf.load_lanes("m", f)
            f.write_text(json.dumps({"models": [{"id": "m", "lanes": [
                dict(lane("A", "reference")), dict(lane("A", "control"))]}]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                mf.load_lanes("m", f)
        mf.load_lanes(mf.DEFAULT_MODEL)  # the committed matrix is valid

    def test_retry_blocked_by_budget_keeps_the_original_error(self):
        calls, slept = [], []

        def post(url, key, body, **_):
            calls.append(1)
            raise http_error(url, 429)
        real_sleep = mf.time.sleep
        mf.time.sleep = slept.append
        try:
            _, rec = mf.call_lane(LANES[1], "k", P[0], 0, None, post=post, budget=mf.Budget(1))
        finally:
            mf.time.sleep = real_sleep
        self.assertEqual(len(calls), 1)
        self.assertEqual(rec["taxonomy"], "rate_limited")
        self.assertIn("HTTP 429", rec["error"])
        self.assertEqual(rec["retry_skipped"], "max_requests")
        self.assertEqual(slept, [])  # budget checked before the backoff

    def test_retry_spends_exactly_one_budget_unit(self):
        n = []

        def post(url, key, body, **_):
            n.append(1)
            if len(n) == 1:
                raise http_error(url, 503)
            return content_shape(["a"])
        budget = mf.Budget(5)
        payload, rec = mf.call_lane(LANES[1], "k", P[0], 0, None, post=post, budget=budget)
        self.assertEqual((rec["status"], rec["http_calls"], budget.used), ("ok", 2, 2))

    def test_plan_log_mentions_retries(self):
        lines = []
        mf.run({"model": "m"}, lanes=LANES, env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"},
               post=two_post(GOOD), log=lines.append)
        self.assertTrue(any("retr" in l and "plan:" in l for l in lines), lines)

    def test_latest_run_dir_ignores_foreign_directories(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("20261014T120000000000001Z", "20261014T130000000000002Z",
                         "zzz-notes", "20261014T999", "latest"):
                (pathlib.Path(d) / name).mkdir()
            (pathlib.Path(d) / "30000101T000000000000000Z.json").write_text("{}")
            self.assertEqual(mf.latest_run_dir(d).name, "20261014T130000000000002Z")

    def test_run_dir_names_sort_in_clock_order_across_a_second(self):
        ticks = iter([1_760_400_000_999_999_999, 1_760_400_001_000_000_001,
                      1_760_400_001_000_000_002])
        real = mf.time.time_ns
        mf.time.time_ns = lambda: next(ticks)
        try:
            names = [mf.new_run_dir("d").name for _ in range(3)]
        finally:
            mf.time.time_ns = real
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(mf.RUN_DIR_RE.match(n) for n in names), names)


def rejects_top_logprobs(url):
    return http_error(url, 400, json.dumps({"error": {
        "message": "top_logprobs must be <= 1 for this model.",
        "type": "invalid_request_error", "param": "top_logprobs"}}).encode())


class StepDownTest(unittest.TestCase):
    """A subject that asks for top_logprobs 5 and is refused steps the same
    probe down to 1 before giving up on logprobs."""

    def test_steps_down_to_one_then_succeeds(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            if body.get("top_logprobs", 0) > 1:
                raise rejects_top_logprobs(url)
            return content_shape(["a"], k=1)
        subj = lane("subj", "subject", "S_KEY", k=5)
        with tempfile.TemporaryDirectory() as d:
            payload, rec = mf.call_lane(subj, "k", P[0], 0, d, post=post)
            files = list(pathlib.Path(d).iterdir())
            # saved under the width actually requested, and found again by load_saved
            self.assertEqual(files, [mf.cache_path(d, subj, P[0], 0, True, 1)])
            got = mf.load_saved([subj], pathlib.Path(d), 1)
        self.assertEqual([b.get("top_logprobs") for b in bodies], [5, 1])
        self.assertEqual((rec["status"], rec["http_calls"], rec["top_logprobs"], rec["stepped_down"]),
                         ("ok", 2, 1, True))
        self.assertTrue(rec["logprobs_requested"])
        self.assertEqual(payload["_fingerprint_top_logprobs"], 1)
        self.assertEqual(got[("subj", P[0], 0)]["_fingerprint_top_logprobs"], 1)

    def test_steps_down_then_falls_back_to_no_logprobs(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            if "logprobs" in body:
                raise rejects_top_logprobs(url)
            return no_logprobs()
        payload, rec = mf.call_lane(lane("subj", "subject", "S_KEY", k=5), "k", P[0], 0, None, post=post)
        self.assertEqual([b.get("top_logprobs") for b in bodies], [5, 1, None])
        self.assertEqual((rec["status"], rec["http_calls"], rec["logprobs_requested"], rec["top_logprobs"]),
                         ("ok", 3, False, 0))
        self.assertFalse(payload["_fingerprint_logprobs_requested"])

    def test_a_subject_asking_for_one_does_not_step_down(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            if "logprobs" in body:
                raise rejects_top_logprobs(url)
            return no_logprobs()
        _, rec = mf.call_lane(lane("subj", "subject", "S_KEY", k=1), "k", P[0], 0, None, post=post)
        self.assertEqual([b.get("top_logprobs") for b in bodies], [1, None])
        self.assertNotIn("stepped_down", rec)

    def test_references_never_step_down(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            raise rejects_top_logprobs(url)
        _, rec = mf.call_lane(lane("refA", "reference", "A_KEY", k=5), "k", P[0], 0, None, post=post)
        self.assertEqual((rec["status"], len(bodies)), ("error", 1))

    def test_step_down_spends_budget_and_is_bounded(self):
        def post(url, key, body, **_):
            if body.get("top_logprobs", 0) > 1:
                raise rejects_top_logprobs(url)
            return content_shape(["a"], k=1)
        budget = mf.Budget(1)
        _, rec = mf.call_lane(lane("subj", "subject", "S_KEY", k=5), "k", P[0], 0, None, post=post, budget=budget)
        self.assertEqual((rec["status"], rec["retry_skipped"], budget.used), ("error", "max_requests", 1))

    def test_served_width_is_reported_and_the_verdict_uses_it(self):
        # The relay caps at 1 today: the whole run steps down, the report
        # says asked 5 / served 1, and the lane is judged on lp_delta.
        def post(url, key, body, **_):
            model = body["model"]
            if model == "subj":
                if body.get("top_logprobs", 0) > 1:
                    raise rejects_top_logprobs(url)
                return content_shape(["a", "b", "c"], k=1, chosen=-0.11)
            if model == "ctrl":
                return content_shape(["q", "r"], prompt_tokens=14)
            return content_shape(["a", "b", "c"], spread=0.04 if model == "refB" else 0.0,
                                 chosen=-0.15 if model == "refB" else -0.1)
        rep = mf.run({"model": "m"}, lanes=LANES, env={"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}, post=post)
        subj = next(l for l in rep["lanes"] if l["name"] == "subj")
        self.assertEqual((subj["topk_requested"], subj["topk_served"], subj["topk"]), (5, 1, 1))
        self.assertEqual(rep["lane_verdicts"]["subj"]["metric"], "lp_delta")
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "same_model")
        self.assertIn("asked 5, served 1", rep["text"])
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])


class RelayBaseTest(unittest.TestCase):
    """--relay-base retargets the configured subject lane; it only appends
    a relay lane when no subject lane exists."""

    def test_retargets_the_existing_subject(self):
        entry = {"id": "deeprelay/m", "request_extra": {"reasoning_effort": "none"}, "max_tokens": 77,
                 "lanes": [lane("deeprelay", "subject", "DEEPRELAY_API_KEY", base_url="https://api.deeprelay.ai/v1",
                                model="deeprelay/m"), lane("refA", "reference"), lane("refB", "reference"),
                           lane("ctrl", "control")]}
        lanes = mf.retarget_relay(mf.lanes_of(entry), "deeprelay/m", "https://api.demo.deeprelay.ai")
        subj = [l for l in lanes if l["role"] in mf.VERDICT_ROLES]
        self.assertEqual(len(subj), 1)
        self.assertEqual(subj[0]["base_url"], "https://api.demo.deeprelay.ai/v1")
        self.assertEqual((subj[0]["name"], subj[0]["model"], subj[0]["top_logprobs"], subj[0]["key_env"],
                          subj[0]["request_extra"], subj[0]["max_tokens"]),
                         ("deeprelay", "deeprelay/m", 5, "DEEPRELAY_API_KEY", {"reasoning_effort": "none"}, 77))
        self.assertEqual(len(lanes), 4)

    def test_appends_a_relay_lane_when_no_subject_exists(self):
        lanes = [lane("refA", "reference"), lane("refB", "reference"), lane("ctrl", "control")]
        out = mf.retarget_relay(lanes, "deeprelay/m", "https://api.deeprelay.ai", request_extra={"x": 1},
                                max_tokens=50)
        self.assertEqual(len(out), 4)
        self.assertEqual((out[-1]["role"], out[-1]["base_url"], out[-1]["model"], out[-1]["top_logprobs"],
                          out[-1]["request_extra"], out[-1]["max_tokens"]),
                         ("relay", "https://api.deeprelay.ai/v1", "deeprelay/m", 1, {"x": 1}, 50))

    def test_run_with_relay_base_sends_the_subject_to_that_base(self):
        urls = []

        def post(url, key, body, **_):
            urls.append(url)
            model = body["model"]
            if model == "ctrl":
                return content_shape(["q", "r"], prompt_tokens=14)
            return content_shape(["a", "b", "c"], spread={"refA": 0.0, "refB": 0.04}.get(model, 0.02),
                                 chosen={"refB": -0.15}.get(model, -0.1))
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "fingerprint_lanes.json"
            f.write_text(json.dumps({"models": [{"id": "deeprelay/m", "lanes": [
                lane("deeprelay", "subject", "DEEPRELAY_API_KEY", base_url="https://api.deeprelay.ai/v1",
                     model="deeprelay/m"),
                lane("refA", "reference"), lane("refB", "reference"), lane("ctrl", "control")]}]}))
            rep = mf.run({"model": "deeprelay/m", "lanes_file": str(f), "relay_base": "https://other.example"},
                         env={"K": "k", "DEEPRELAY_API_KEY": "k"}, post=post)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertEqual(list(rep["lane_verdicts"]), ["deeprelay"])
        self.assertTrue(any(u.startswith("https://other.example/v1/") for u in urls))
        self.assertFalse(any("api.deeprelay.ai" in u for u in urls))


class CliTest(unittest.TestCase):
    def test_single_model_cli_exits_zero_only_on_pass(self):
        real_run, real_stdout = mf.run, sys.stdout
        sys.stdout = io.StringIO()
        try:
            for verdict, code in (("pass", 0), ("inconclusive", 1), ("mismatch", 1), ("insufficient", 1)):
                mf.run = lambda config, env=None, log=None, v=verdict: {"verdict": v, "text": f"VERDICT: {v}"}
                with self.assertRaises(SystemExit) as cm:
                    mf.main(["--model", "deeprelay/m"])
                self.assertEqual(cm.exception.code, code, verdict)
        finally:
            mf.run, sys.stdout = real_run, real_stdout


class CallLaneTest(unittest.TestCase):
    def test_subject_falls_back_only_on_a_logprobs_rejection(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            if "logprobs" in body:
                raise http_error(url, 400, json.dumps({"error": {
                    "message": "top_logprobs must be <= 1 for deeprelay/m.",
                    "type": "invalid_request_error", "param": "top_logprobs"}}).encode())
            return no_logprobs()
        payload, rec = mf.call_lane(mf.relay_lane("m", "https://api.example"), "k",
                                    "p01_capital", 0, None, post=post)
        self.assertEqual(rec["status"], "ok")
        self.assertFalse(rec["logprobs_requested"])
        self.assertFalse(payload["_fingerprint_logprobs_requested"])
        self.assertEqual(rec["http_calls"], 2)

    def test_subject_does_not_fall_back_on_other_400s(self):
        calls = []

        def post(url, key, body, **_):
            calls.append(body)
            raise http_error(url, 400, json.dumps({"error": {
                "message": "model not found", "type": "invalid_request_error",
                "param": "model"}}).encode())
        _, rec = mf.call_lane(mf.relay_lane("m", "https://api.example"), "k",
                              "p01_capital", 0, None, post=post)
        self.assertEqual(rec["status"], "error")
        self.assertEqual(len(calls), 1)

    def test_reference_never_falls_back(self):
        calls = []

        def post(url, key, body, **_):
            calls.append(body)
            raise http_error(url, 400, json.dumps({"error": {"param": "logprobs", "message": "no"}}).encode())
        _, rec = mf.call_lane(LANES[1], "k", "p01_capital", 0, None, post=post)
        self.assertEqual((rec["status"], len(calls)), ("error", 1))

    def test_http_error_without_a_body_is_handled(self):
        def post(url, key, body, **_):
            raise urllib.error.HTTPError(url, 503, "down", {}, None)
        _, rec = mf.call_lane(LANES[1], "k", "p01_capital", 0, None, post=post)
        self.assertEqual(rec["taxonomy"], "http_5xx")

    def test_saved_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            mf.call_lane(LANES[1], "k", "p01_capital", 0, d,
                         post=lambda *a, **k: content_shape(["a"]))
            files = list(pathlib.Path(d).iterdir())
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].name.endswith(".json"))

    def test_relay_base_normalizes_v1(self):
        self.assertEqual(mf.relay_lane("m", "https://x")["base_url"], "https://x/v1")
        self.assertEqual(mf.relay_lane("m", "https://x/v1/")["base_url"], "https://x/v1")

    def test_keys_never_appear_in_records_or_reports(self):
        secret = "sk-SECRET-VALUE-123"

        def post(url, key, body, **_):
            raise http_error(url, 503, b"server says " + key.encode())
        rep = mf.run({"model": "m"}, lanes=LANES,
                     env={"S_KEY": secret, "A_KEY": secret, "B_KEY": secret}, post=post)
        # The error body may echo whatever the server sent; the harness
        # itself never adds the key to a record field other than the body.
        self.assertNotIn(secret, json.dumps({k: v for k, v in rep.items() if k != "requests"}))


class ConfigTest(unittest.TestCase):
    def test_committed_matrix_is_well_formed(self):
        cfg = mf.load_matrix()
        self.assertEqual(cfg["schema_version"], 1)
        entries = mf.model_entries(cfg)
        self.assertEqual(len(entries), 5)
        self.assertNotIn("deeprelay/kimi-k3", entries)  # the relay pins its temperature to 1.0
        for model, entry in entries.items():
            self.assertTrue(model.startswith("deeprelay/"), model)
            self.assertTrue(entry.get("display_name") and entry.get("weights"), model)
            lanes = mf.lanes_of(entry)
            roles = [l["role"] for l in lanes]
            self.assertEqual(roles.count("subject"), 1, model)
            self.assertGreaterEqual(roles.count("reference"), 2, model)
            # At least one reference must carry logprobs; text lanes inform only.
            self.assertTrue(any(l["role"] == "reference" and not mf.is_text(l) for l in lanes), model)
            for l in lanes:
                if mf.is_text(l):
                    self.assertNotIn("top_logprobs", l, l)
            self.assertGreaterEqual(roles.count("control"), 1, model)
            subj = next(l for l in lanes if l["role"] == "subject")
            self.assertEqual(subj["base_url"], "https://api.deeprelay.ai/v1", model)
            self.assertEqual(subj["model"], model)
            self.assertEqual(subj["key_env"], "DEEPRELAY_API_KEY")
            self.assertEqual(subj["top_logprobs"], 5, model)
            for l in entry["lanes"]:
                self.assertEqual(set(l) - set(mf.LANE_KEYS) - set(mf.LANE_OPTIONAL_KEYS), set(), l)
                self.assertTrue(l["key_env"].isupper(), l)
                self.assertFalse(l["key_env"].startswith("sk"), l)
                self.assertNotIn("gpu.ai", json.dumps(l))

    def test_subject_asks_the_relay_for_five(self):
        for model in ("deeprelay/deepseek-v4.1-flash", "deeprelay/glm-5.2", "deeprelay/glm-5.3",
                      "deeprelay/glm-5.3-flash", "deeprelay/qwen3.8-flash"):
            subj = next(l for l in mf.load_lanes(model) if l["role"] == "subject")
            self.assertEqual(subj["top_logprobs"], 5, model)

    def test_matrix_under_another_name_with_model_selection(self):
        # The vendored copy ships matrix.json as fingerprint_lanes.json and
        # passes --lanes-file plus --model.
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "fingerprint_lanes.json"
            f.write_text(mf.DEFAULT_LANES_FILE.read_text())
            lanes = mf.load_lanes("deeprelay/deepseek-v4.1-flash", f)
            self.assertEqual([l["role"] for l in lanes][:3], ["subject", "reference", "reference"])
            self.assertEqual(lanes[0]["request_extra"], {"reasoning_effort": "none"})
            with self.assertRaises(KeyError):
                mf.load_lanes("deeprelay/kimi-k3", f)

    def test_thinking_only_models_keep_thinking_on_with_a_large_budget(self):
        for model in ("deeprelay/glm-5.3", "deeprelay/glm-5.3-flash"):
            entry = mf.load_model_entry(model)
            self.assertEqual(entry["max_tokens"], 2000)
            self.assertNotIn("reasoning_effort", entry.get("request_extra") or {})
        for model in ("deeprelay/deepseek-v4.1-flash", "deeprelay/glm-5.2", "deeprelay/qwen3.8-flash"):
            self.assertEqual(mf.load_model_entry(model)["request_extra"], {"reasoning_effort": "none"})

    def test_prompts_keep_invisible_characters(self):
        self.assertIn("​", mf.PROMPTS["p03_tokenizer"])
        self.assertIn("\xa0", mf.PROMPTS["p03_tokenizer"])
        self.assertEqual(len(mf.PROMPTS), 12)


class TextLaneTest(unittest.TestCase):
    """A lane whose API offers no logprobs (a vendor's own endpoint) is
    probed at temperature 0, compared on content and signature, reported,
    and never judges."""
    TEXT = {"name": "vendor", "role": "reference", "evidence": "text",
            "base_url": "https://vendor.example/v1", "model": "m", "key_env": "V_KEY"}

    def test_validation(self):
        mf.validate_lanes(LANES + [self.TEXT])
        with self.assertRaises(ValueError):
            mf.validate_lanes(LANES + [dict(self.TEXT, evidence="vibes")])
        with self.assertRaises(ValueError):
            mf.validate_lanes([dict(l, evidence="text") if l["role"] == "subject" else l for l in LANES])

    def test_probe_never_asks_for_logprobs(self):
        bodies = []

        def post(url, key, body, **_):
            bodies.append(body)
            return no_logprobs(prompt_tokens=9, content="ab")
        payload, rec = mf.call_lane(dict(self.TEXT, top_logprobs=5), "k", "p01_capital", 0, None, post=post)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["evidence"], "text")
        self.assertNotIn("logprobs", bodies[0])
        self.assertNotIn("top_logprobs", bodies[0])
        self.assertEqual(bodies[0]["temperature"], 0)
        self.assertFalse(payload["_fingerprint_logprobs_requested"])

    def test_reported_but_never_counted_or_judging(self):
        lanes = LANES + [self.TEXT]
        r = baseline()
        fill(r, "vendor", lambda i, rep: no_logprobs(content="abc" if i != 3 else "z"))
        rep = mf.analyze(lanes, r)
        self.assertEqual(rep["verdict"], "pass")
        t = rep["text_references"][0]
        self.assertEqual(t["lane"], "vendor")
        self.assertEqual(t["vs"]["subj"]["role"], "subject")
        self.assertEqual(t["vs"]["subj"]["same_content"], len(mf.PROMPTS) - 1)
        self.assertEqual(t["self_identical"], t["self_pairs"])
        self.assertTrue(any(n.startswith("text lane vendor") for n in rep["notes"]), rep["notes"])
        self.assertNotIn("vendor", rep["lane_verdicts"])
        self.assertEqual(next(l for l in rep["lanes"] if l["name"] == "vendor")["evidence"], "text")
        self.assertEqual(next(l for l in rep["lanes"] if l["name"] == "vendor")["fallback_prompts"], 0)
        # Without the two logprob references the text lane does not fill in.
        only_one = [l for l in lanes if l["name"] != "refB"]
        r2 = {k: v for k, v in r.items() if k[0] != "refB"}
        rep2 = mf.analyze(only_one, r2)
        self.assertEqual(rep2["verdict"], "insufficient")
        self.assertTrue(any("got 1" in x for x in rep2["reasons"]), rep2["reasons"])
        self.assertEqual(rep2["text_references"][0]["lane"], "vendor")
        self.assertTrue(rep2["notes"])

    def test_render_has_a_text_section(self):
        lanes = LANES + [self.TEXT]
        r = baseline()
        fill(r, "vendor", lambda i, rep: no_logprobs(content="abc"))
        rep = mf.analyze(lanes, r)
        self.assertIn("4b. TEXT LANES", mf.render(rep, r))


if __name__ == "__main__":
    unittest.main()
