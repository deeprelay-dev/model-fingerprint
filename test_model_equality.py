"""Hermetic tests for model_equality.py: the sampling request, the
collection rules, and the decision rule applied to the published package's
statistics. No network; samples come from a fake post or are handed to the
judge directly. The package (torch) must be installed: see requirements.txt."""

import json
import pathlib
import random
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_equality as me  # noqa: E402
import model_fingerprint as mf  # noqa: E402

mf.RETRY_BACKOFF_S = 0.0

# Small, fast settings for the judge: the rule is what is under test, not
# the package's power.
SETTINGS = {"per_prompt": 6, "max_tokens": 50, "L": 40, "permutations": 20, "alpha": 0.05,
            "workers": 2, "request_extra": {}, "enabled": True}


def lane(name, role, key_env="K", model=None, **extra):
    d = {"name": name, "role": role, "base_url": f"https://{name}.example/v1",
         "model": model or f"org/{name}", "key_env": key_env, "request_extra": {}}
    d.update(extra)
    return d


ENTRY = {"id": "deeprelay/m", "weights": "org/M", "request_extra": {"reasoning_effort": "none"},
         "lanes": [lane("subj", "subject"), lane("refA", "reference"), lane("refB", "reference"),
                   lane("zai", "reference", evidence="text"), lane("ctrl", "control")]}
LANES = ENTRY["lanes"]


def family(seed, vocab):
    """A deterministic text generator: a 'model' whose completions for a
    prompt are short word sequences drawn from its own vocabulary, so two
    lanes of the same family give the same distribution and a lane of
    another family a visibly different one."""
    def gen(pid, rep, salt=""):
        rng = random.Random(f"{seed}:{pid}:{rep}:{salt}")
        return " ".join(rng.choice(vocab) for _ in range(8))
    return gen


FAM_A = family("A", ["the", "harbour", "stone", "quay", "tide", "boats", "fish", "salt", "wind", "rope"])
FAM_B = family("B", ["der", "Mühle", "Wasser", "Bach", "Herbst", "Stein", "Korn", "Rad", "Fluss", "Tal"])


def fake_post(texts_for):
    """texts_for(lane_model, pid, rep) -> text or an exception to raise."""
    def post(url, key, body, timeout=None):
        pid = next(p for p, t in me.SAMPLE_PROMPTS.items() if t == body["messages"][0]["content"])
        rep = post.calls.setdefault((body["model"], pid), 0)
        post.calls[(body["model"], pid)] = rep + 1
        out = texts_for(body["model"], pid, rep)
        if isinstance(out, Exception):
            raise out
        return {"id": "x", "choices": [{"message": {"role": "assistant", "content": out},
                                        "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8}}
    post.calls = {}
    return post


def samples_from(lanes, gen_for, per_prompt=SETTINGS["per_prompt"], salt=""):
    """Samples dict as collect() returns it, without HTTP."""
    out = {}
    for l in lanes:
        g = gen_for(l["name"])
        if g is None:
            continue
        for pid in me.PROMPT_IDS:
            for rep in range(per_prompt):
                out[(l["name"], pid, rep)] = g(pid, rep, salt + l["name"])
    return out


def http_error(code, body=b"{}"):
    return urllib.error.HTTPError("https://x", code, "err", {}, __import__("io").BytesIO(body))


class RequestTest(unittest.TestCase):
    def test_body_samples_at_temperature_one_without_logprobs(self):
        body = me.build_body(lane("refA", "reference"), "hello", 77, {"reasoning_effort": "low"})
        self.assertEqual((body["temperature"], body["top_p"], body["max_tokens"], body["stream"]),
                         (1, 1, 77, False))
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("logprobs", body)

    def test_stream_lane_samples_over_sse(self):
        self.assertTrue(me.build_body(lane("t", "reference", stream=True), "p", 10, {})["stream"])

    def test_lane_overrides_are_what_differs_from_the_entry(self):
        entry = {"request_extra": {"reasoning_effort": "none"}}
        folded = mf.lanes_of({"request_extra": entry["request_extra"], "lanes": [
            lane("a", "control"), lane("b", "control", request_extra={"reasoning_effort": "low"})]})
        self.assertEqual(me.lane_overrides(folded[0], entry), {})
        self.assertEqual(me.lane_overrides(folded[1], entry), {"reasoning_effort": "low"})
        body = me.build_body(folded[1], "p", 10, {"reasoning_effort": "none"},
                             me.lane_overrides(folded[1], entry))
        self.assertEqual(body["reasoning_effort"], "low")

    def test_samples_extra_wins(self):
        body = me.build_body(lane("z", "reference", samples_extra={"thinking": {"type": "disabled"}}),
                             "p", 10, {"reasoning_effort": "none"})
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_settings_merge_in_order(self):
        entry = {"request_extra": {"reasoning_effort": "none"},
                 "samples": {"per_prompt": 3, "request_extra": {"reasoning_effort": "low"}}}
        s = me.settings_for(entry, {"samples": {"per_prompt": 9, "max_tokens": 500}})
        self.assertEqual((s["per_prompt"], s["max_tokens"], s["L"], s["request_extra"]),
                         (3, 500, me.L, {"reasoning_effort": "low"}))
        s = me.settings_for({"request_extra": {"reasoning_effort": "none"}}, {})
        self.assertEqual((s["per_prompt"], s["request_extra"], s["deadline_s"], s["max_requests"]),
                         (me.PER_PROMPT, {"reasoning_effort": "none"}, me.DEADLINE_S, None))

    def test_prompts_ask_not_to_repeat_and_cover_several_languages(self):
        self.assertGreaterEqual(len(me.PROMPT_IDS), 12)
        self.assertEqual(len(set(me.SAMPLE_PROMPTS.values())), len(me.PROMPT_IDS))
        self.assertTrue(any("_zh" in p for p in me.PROMPT_IDS))
        self.assertTrue(any("_ru" in p for p in me.PROMPT_IDS))


class CollectTest(unittest.TestCase):
    def test_collects_every_sample_and_saves_raw(self):
        post = fake_post(lambda model, pid, rep: FAM_A(pid, rep, model))
        with tempfile.TemporaryDirectory() as d:
            samples, records, skipped = me.collect(LANES, {"K": "k"}, d, SETTINGS, 10_000, post=post)
            self.assertEqual(len(samples), len(LANES) * len(me.PROMPT_IDS) * SETTINGS["per_prompt"])
            self.assertEqual(skipped, [])
            self.assertTrue(all(r["status"] == "ok" for r in records))
            self.assertTrue(me.sample_path(d, LANES[0], me.PROMPT_IDS[0], 0).is_file())
            again = me.load_saved(LANES, d, SETTINGS["per_prompt"])
            self.assertEqual(again, samples)

    def test_missing_key_skips_the_lane(self):
        post = fake_post(lambda model, pid, rep: FAM_A(pid, rep))
        lanes = [lane("subj", "subject"), lane("refA", "reference", key_env="MISSING")]
        samples, records, skipped = me.collect(lanes, {"K": "k"}, None, SETTINGS, 10_000, post=post)
        self.assertEqual([s["lane"] for s in skipped], ["refA"])
        self.assertTrue(all(k[0] == "subj" for k in samples))

    def test_transient_errors_are_retried_and_empty_content_is_an_error(self):
        def texts(model, pid, rep):
            if model == "org/refA" and rep == 0 and pid == me.PROMPT_IDS[0]:
                return http_error(503)
            if model == "org/refB":
                return ""
            return FAM_A(pid, rep)
        post = fake_post(texts)
        lanes = [lane("refA", "reference"), lane("refB", "reference")]
        samples, records, _ = me.collect(lanes, {"K": "k"}, None, dict(SETTINGS, per_prompt=1), 10_000,
                                         post=post)
        a = next(r for r in records if r["lane"] == "refA" and r["prompt"] == me.PROMPT_IDS[0])
        self.assertEqual((a["status"], a["retries"], a["http_calls"]), ("ok", 1, 2))
        b = [r for r in records if r["lane"] == "refB"]
        self.assertTrue(all(r["status"] == "error" and r["taxonomy"] == "empty_content" for r in b))
        self.assertFalse(any(k[0] == "refB" for k in samples))

    def test_budget_and_deadline_stop_the_sweep(self):
        post = fake_post(lambda model, pid, rep: FAM_A(pid, rep))
        samples, records, _ = me.collect([lane("refA", "reference")], {"K": "k"}, None, SETTINGS, 5,
                                         post=post)
        self.assertEqual(len(samples), 5)
        self.assertTrue(any(r.get("taxonomy") == "max_requests" for r in records))
        samples, records, _ = me.collect([lane("refA", "reference")], {"K": "k"}, None, SETTINGS, 10_000,
                                         post=post, deadline_s=0)
        self.assertEqual(len(samples), 0)
        self.assertTrue(all(r.get("taxonomy") == "deadline" for r in records))


class JudgeTest(unittest.TestCase):
    def judge(self, gen_for, lanes=LANES, settings=SETTINGS, **kw):
        return me.judge(lanes, samples_from(lanes, gen_for), settings, **kw)

    def test_same_family_passes_and_the_control_separates(self):
        rep = self.judge(lambda n: FAM_B if n == "ctrl" else FAM_A)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        lv = rep["lane_verdicts"]["subj"]
        self.assertEqual(lv["verdict"], "same_model")
        self.assertEqual(set(lv["references"]), {"refA", "refB", "zai"})
        self.assertTrue(all(v["within"] for v in lv["references"].values()))
        self.assertEqual(rep["band"]["references"], ["refA", "refB", "zai"])
        c = rep["controls"][0]
        self.assertTrue(c["separates"])
        self.assertLess(c["subjects"]["subj"]["p"], 0.05)
        self.assertGreater(c["min_reference_mmd"], rep["band"]["mmd"])
        self.assertEqual(rep["statistic"], "mmd_hamming")
        self.assertTrue(rep["package"].startswith("model-equality-testing"))
        kinds = {(p["a"], p["b"]): p["kind"] for p in rep["pairs"]}
        self.assertEqual(kinds[("subj", "refA")], "subject-reference")
        self.assertEqual(kinds[("refA", "refB")], "reference-reference")
        self.assertEqual(kinds[("subj", "ctrl")], "subject-control")
        self.assertIsNone(next(p for p in rep["pairs"] if p["kind"] == "reference-reference")["p"])
        self.assertIsNotNone(next(p for p in rep["pairs"] if p["kind"] == "subject-reference")["p"])
        self.assertIn("SAME_MODEL", me.render(rep))

    def test_a_different_model_in_the_subject_lane_is_a_mismatch(self):
        rep = self.judge(lambda n: FAM_B if n in ("subj", "ctrl") else FAM_A)
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "mismatch")
        self.assertEqual(rep["verdict"], "mismatch")

    def test_a_control_that_does_not_separate_makes_the_run_inconclusive(self):
        rep = self.judge(lambda n: FAM_A)
        self.assertEqual(rep["lane_verdicts"]["subj"]["verdict"], "same_model")
        self.assertFalse(rep["controls"][0]["separates"])
        self.assertEqual(rep["verdict"], "inconclusive")
        self.assertTrue(any("no gating control separates" in r for r in rep["reasons"]))

    def test_an_advisory_control_only_notes(self):
        lanes = [lane("subj", "subject"), lane("refA", "reference"), lane("refB", "reference"),
                 lane("ctrl", "control"), lane("adv", "control", advisory=True)]
        rep = self.judge(lambda n: FAM_B if n == "ctrl" else FAM_A, lanes=lanes)
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertTrue(any("adv" in n and "advisory" in n for n in rep["notes"]))

    def test_fewer_than_two_references_is_insufficient(self):
        lanes = [lane("subj", "subject"), lane("refA", "reference"), lane("ctrl", "control")]
        rep = self.judge(lambda n: FAM_B if n == "ctrl" else FAM_A, lanes=lanes)
        self.assertEqual(rep["verdict"], "insufficient")
        self.assertIn("1 usable reference", rep["reasons"][0])
        self.assertEqual(rep["lane_verdicts"], {})

    def test_a_thin_lane_is_dropped(self):
        samples = samples_from(LANES, lambda n: FAM_B if n == "ctrl" else FAM_A)
        # refB keeps 5 of 6 on every prompt (above the 80% floor) but only 2 on one
        for k in [k for k in samples if k[0] == "refB" and (k[2] >= 5 or (k[1] == me.PROMPT_IDS[0] and k[2] >= 2))]:
            del samples[k]
        rep = me.judge(LANES, samples, SETTINGS)
        self.assertEqual([d["lane"] for d in rep["dropped"]], ["refB"])
        self.assertIn("under 5 samples", rep["dropped"][0]["reason"])
        self.assertEqual(rep["band"]["references"], ["refA", "zai"])
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])
        self.assertIn("DROPPED", me.render(rep))

    def test_no_subject_samples_is_insufficient(self):
        samples = samples_from(LANES, lambda n: None if n == "subj" else FAM_A)
        rep = me.judge(LANES, samples, SETTINGS)
        self.assertEqual(rep["verdict"], "insufficient")

    def test_skipped_lanes_are_listed_not_judged(self):
        samples = samples_from(LANES, lambda n: None if n == "zai" else (FAM_B if n == "ctrl" else FAM_A))
        rep = me.judge(LANES, samples, SETTINGS, skipped=[{"lane": "zai", "reason": "ZAI_API_KEY is not set"}])
        self.assertEqual(rep["skipped"], ["zai"])
        self.assertNotIn("zai", [l["name"] for l in rep["lanes"]])
        self.assertEqual(rep["verdict"], "pass", rep["reasons"])

    def test_judge_is_deterministic_given_the_seed(self):
        a = self.judge(lambda n: FAM_B if n == "ctrl" else FAM_A, seed=3)
        b = self.judge(lambda n: FAM_B if n == "ctrl" else FAM_A, seed=3)
        self.assertEqual(a["pairs"], b["pairs"])


class RunTest(unittest.TestCase):
    def test_run_samples_judges_and_never_raises(self):
        post = fake_post(lambda model, pid, rep: FAM_B(pid, rep) if model == "org/ctrl" else FAM_A(pid, rep))
        with tempfile.TemporaryDirectory() as d:
            rep = me.run(ENTRY, LANES, {"K": "k"}, SETTINGS, raw_dir=d, post=post)
            self.assertEqual(rep["verdict"], "pass", rep["reasons"])
            self.assertEqual(rep["source"]["mode"], "fresh")
            self.assertFalse(any(r.get("taxonomy") == "max_requests" for r in rep["request_errors"]))
            self.assertEqual(rep["requests"], len(LANES) * len(me.PROMPT_IDS) * SETTINGS["per_prompt"])
            self.assertEqual(rep["request"]["max_tokens"], 50)
            self.assertIn("verdict PASS", rep["text"])
            again = me.run(ENTRY, LANES, {}, SETTINGS, raw_dir=d, analyze_only=True)
            self.assertEqual(again["verdict"], "pass", again["reasons"])
            self.assertEqual(again["source"]["mode"], "analyze_only")

    def test_a_tight_budget_in_the_settings_is_honoured(self):
        post = fake_post(lambda model, pid, rep: FAM_A(pid, rep))
        rep = me.run(ENTRY, LANES, {"K": "k"}, dict(SETTINGS, max_requests=9), post=post)
        self.assertEqual(rep["requests"], 9)
        self.assertEqual(rep["verdict"], "insufficient")

    def test_a_failure_becomes_an_error_report(self):
        rep = me.run(ENTRY, LANES, {}, SETTINGS, raw_dir=None, analyze_only=True)
        self.assertEqual(rep["verdict"], "error")
        self.assertIn("analyze_only needs raw_dir", rep["reasons"][0])
        self.assertIn("ERROR", rep["text"])

    def test_json_serialisable(self):
        post = fake_post(lambda model, pid, rep: FAM_B(pid, rep) if model == "org/ctrl" else FAM_A(pid, rep))
        rep = me.run(ENTRY, LANES, {"K": "k"}, SETTINGS, post=post)
        json.dumps(rep)


if __name__ == "__main__":
    unittest.main()
