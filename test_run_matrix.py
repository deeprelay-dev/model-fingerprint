"""Hermetic tests for run_matrix.py and publish.py: the bench-schema report
the deeprelay server ingests, the exit-code contract, provenance, and the
publisher's retry rules. No network."""

import io
import json
import pathlib
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_equality as me  # noqa: E402
import model_fingerprint as mf  # noqa: E402
import publish  # noqa: E402
import run_matrix as rm  # noqa: E402
from test_model_fingerprint import LANES, baseline, content_shape, fill, lane, no_logprobs  # noqa: E402

mf.RETRY_BACKOFF_S = 0.0

ENTRY = {"id": "deeprelay/m", "display_name": "M", "weights": "org/M",
         "request_extra": {"reasoning_effort": "none"}, "max_tokens": 300,
         "lanes": [lane("subj", "subject", "S_KEY", base_url="https://api.deeprelay.ai/v1"),
                   lane("refA", "reference", "A_KEY"), lane("refB", "reference", "B_KEY"),
                   lane("ctrl", "control", "A_KEY")]}


def matrix_file(d, entries):
    f = pathlib.Path(d) / "matrix.json"
    f.write_text(json.dumps({"schema_version": 1, "defaults": {"replicates": 2}, "models": entries}))
    return str(f)


def canned_samples(verdict="pass", lane_verdict=None, judged=("refA", "refB"), reasons=()):
    """A stand-in for me.run: the sampling test's report with one subject."""
    lane_verdict = lane_verdict or {"pass": "same_model", "mismatch": "mismatch",
                                    "inconclusive": "inconclusive"}.get(verdict, "insufficient")

    def run_samples(entry, lanes, env, settings, **kw):
        run_samples.calls.append((entry["id"], settings, kw))
        subj = next((l["name"] for l in lanes if l["role"] in mf.VERDICT_ROLES), "subj")
        return {"method": "model_equality_testing", "verdict": verdict, "reasons": list(reasons),
                "lanes": [{"name": l["name"], "role": l["role"], "model": l["model"], "samples": 90,
                           "planned": 90, **({} if l["name"] in judged or l["role"] in mf.VERDICT_ROLES
                                            else {"dropped": "0/90 samples"})} for l in lanes],
                "lane_verdicts": ({subj: {"verdict": lane_verdict, "reason": "canned", "distance": 0.02,
                                          "p": 0.3, "band": 0.05, "references": {}}}
                                  if verdict not in ("insufficient", "error") else {}),
                "controls": [{"lane": "ctrl", "separates": verdict == "pass", "advisory": False}],
                "pairs": [], "dropped": [], "notes": [],
                "requests": 0 if verdict in ("error", "insufficient") else 450, "request_errors": [],
                "text": "SAMPLES text"}
    run_samples.calls = []
    return run_samples


def fake_run(responses_for):
    """A stand-in for mf.run that analyzes canned responses per model."""
    def run_model(config, lanes=None, env=None, log=None, **_):
        r = responses_for(config["model"])
        if r is None:
            return {"verdict": "error", "reasons": ["boom"], "requests": {"sent": 0, "errors": []},
                    "model": config["model"], "text": ""}
        rep = mf.analyze(lanes, r, config.get("replicates", 2))
        rep["requests"] = {"sent": 7, "errors": [{"taxonomy": "rate_limited"}]}
        rep["text"] = "text"
        rep["model"] = config["model"]
        return rep
    return run_model


class ReportShapeTest(unittest.TestCase):
    def run_one(self, responses, entries=None, models=None, env=None, samples="pass", extra_argv=()):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            summ = pathlib.Path(d) / "summary.md"
            argv = ["--matrix", matrix_file(d, entries or [ENTRY]), "--json-out", str(out),
                    "--summary-out", str(summ), "--quiet", *extra_argv]
            if models:
                argv += ["--models", models]
            run_samples = (samples if callable(samples) else
                           canned_samples(samples) if samples else canned_samples())
            code = rm.main(argv, env=env or {}, run_model=fake_run(lambda m: responses),
                           run_samples=run_samples)
            return code, json.loads(out.read_text()), summ.read_text()

    def test_go_wire_contract(self):
        code, rep, md = self.run_one(baseline())
        self.assertEqual(code, rm.EXIT_OK)
        self.assertEqual(rep["schema_version"], 1)
        self.assertEqual(rep["harness_version"], mf.HARNESS_VERSION)
        for k in ("id", "url", "attempt", "repository", "workflow", "commit_sha", "trigger",
                  "started_at", "finished_at", "duration_ms"):
            self.assertIn(k, rep["run"])
        self.assertTrue(rep["run"]["id"].startswith("local-"))
        self.assertTrue(rep["run"]["started_at"].endswith("Z"))
        s = rep["settings"]
        self.assertEqual((s["skip_perf"], s["skip_fingerprint"], s["fingerprint_prompts"],
                          s["fingerprint_replicates"], s["fingerprint_max_tokens"], s["perf_trials"],
                          s["perf_prompts"]), (True, False, 12, 2, 300, 0, []))
        m = rep["models"][0]
        self.assertEqual((m["id"], m["display_name"], m["weights"]), ("deeprelay/m", "M", "org/M"))
        self.assertEqual([l["role"] for l in m["lanes"]], ["subject", "reference", "reference"])
        self.assertEqual(m["lanes"][0]["host"], "api.deeprelay.ai")
        self.assertEqual(m["lanes"][0]["weights_verdict"], "same_model")
        self.assertEqual(m["lanes"][1]["weights_verdict"], "reference")
        fp = m["fingerprint"]
        self.assertEqual(fp["verdict"], "pass")
        self.assertNotIn("text", fp)
        self.assertIsInstance(fp["requests"], int)  # Go: Requests int
        self.assertEqual(fp["requests"], 7 + 450)  # logprob probes + samples
        self.assertTrue(all(isinstance(x, str) for x in fp["skipped"]))  # Go: []string
        self.assertIsInstance(fp["band"]["jsd"], float)  # Go: map[string]*float64
        self.assertIn("noise_cap_x", fp["thresholds"])
        for l in fp["lanes"]:
            self.assertTrue(all(x is None or isinstance(x, int) for x in l["signature"]))
        for v in fp["lane_verdicts"].values():
            self.assertIsInstance(v["aligned"], dict)
            self.assertIsInstance(v["noise_capped"], bool)
        for c in fp["controls"]:
            self.assertIsInstance(c["by_metric"], dict)
        self.assertEqual(m["skipped"], {})
        self.assertEqual(rep["summary"]["models"], 1)
        self.assertEqual(rep["summary"]["lanes"], 3)
        self.assertEqual(rep["summary"]["requests"], 7 + 450)
        self.assertEqual(rep["summary"]["verdicts"], {"pass": 1})
        self.assertEqual(rep["summary"]["errors"], {"rate_limited": 1})
        self.assertEqual(fp["method"], "model_equality_testing")
        self.assertEqual(fp["logprobs_verdict"], "pass")
        self.assertEqual(fp["samples"]["verdict"], "pass")
        self.assertNotIn("text", fp["samples"])
        self.assertEqual(s["samples_method"], "model_equality_testing")
        self.assertEqual(s["samples_prompts"], len(me.PROMPT_IDS))
        self.assertIn("| M | **pass** | pass |", md)


    def test_github_provenance(self):
        env = {"GITHUB_RUN_ID": "123", "GITHUB_SERVER_URL": "https://github.com",
               "GITHUB_REPOSITORY": "deeprelay-dev/model-fingerprint", "GITHUB_RUN_ATTEMPT": "2",
               "GITHUB_WORKFLOW": "verify", "GITHUB_SHA": "abc", "GITHUB_EVENT_NAME": "schedule"}
        _, rep, _ = self.run_one(baseline(), env=env)
        self.assertEqual(rep["run"]["id"], "123")
        self.assertEqual(rep["run"]["url"], "https://github.com/deeprelay-dev/model-fingerprint/actions/runs/123")
        self.assertEqual((rep["run"]["attempt"], rep["run"]["workflow"], rep["run"]["commit_sha"],
                          rep["run"]["trigger"], rep["run"]["repository"]),
                         (2, "verify", "abc", "schedule", "deeprelay-dev/model-fingerprint"))

    def test_mismatch_exits_2(self):
        r = baseline(subject=lambda i, rep: content_shape(["a", "b", "c"], prompt_tokens=14 + i))
        code, rep, _ = self.run_one(r)
        self.assertEqual(code, rm.EXIT_MISMATCH)
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "mismatch")

    def test_inconclusive_counts_as_judged(self):
        lanes = [l for l in ENTRY["lanes"] if l["role"] != "control"]
        r = {k: v for k, v in baseline().items() if k[0] != "ctrl"}
        code, rep, _ = self.run_one(r, entries=[dict(ENTRY, lanes=lanes)],
                                    samples=canned_samples("inconclusive", lane_verdict="same_model"))
        self.assertEqual(code, rm.EXIT_OK)
        # The lane matched; the RUN is inconclusive because no control
        # separated. The server marks a model verified only when both the
        # fingerprint verdict is pass and the subject lane is same_model.
        self.assertEqual(rep["models"][0]["fingerprint"]["verdict"], "inconclusive")
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "same_model")

    def test_nothing_judged_exits_1(self):
        r = {k: v for k, v in baseline().items() if k[0] != "subj"}
        code, rep, _ = self.run_one(r, samples="insufficient")
        self.assertEqual(code, rm.EXIT_NOTHING_JUDGED)
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "insufficient")
        self.assertEqual(rep["models"][0]["fingerprint"]["verdict"], "insufficient")

    def test_error_model_still_produces_a_decodable_row(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            code = rm.main(["--matrix", matrix_file(d, [ENTRY]), "--json-out", str(out), "--quiet"],
                           env={}, run_model=fake_run(lambda m: None), run_samples=canned_samples("error"))
            rep = json.loads(out.read_text())
        self.assertEqual(code, rm.EXIT_NOTHING_JUDGED)
        fp = rep["models"][0]["fingerprint"]
        self.assertEqual(fp["verdict"], "error")
        self.assertEqual(fp["requests"], 0)
        for k in ("lanes", "skipped", "noise", "pairs", "lane_verdicts", "controls", "prompts", "thresholds"):
            self.assertIn(k, fp)
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "error")

    def test_skipped_lanes_land_in_the_model_map(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            code = rm.main(["--matrix", matrix_file(d, [ENTRY]), "--json-out", str(out), "--quiet"],
                           env={"A_KEY": "k", "B_KEY": "k"},
                           run_model=lambda config, lanes, env, log: mf.run(
                               config, lanes=lanes, env=env, log=log,
                               post=lambda u, k, b, **_: content_shape(["a"], prompt_tokens=99 if "ctrl" in u else 40)),
                           run_samples=canned_samples("insufficient"))
            rep = json.loads(out.read_text())
        self.assertEqual(code, rm.EXIT_NOTHING_JUDGED)
        self.assertEqual(rep["models"][0]["skipped"], {"subj": "S_KEY is not set"})
        self.assertEqual(rep["models"][0]["fingerprint"]["skipped"], ["subj"])

    def test_analyze_only_rejudges_saved_responses_without_sending(self):
        sent = []

        def post(url, key, body, **_):
            sent.append(1)
            return content_shape(["a"], prompt_tokens=99 if "ctrl" in url else 40,
                                 chosen=-0.1 - {"refB": 0.02, "subj": 0.01}.get(body["model"], 0.0))
        env = {"S_KEY": "k", "A_KEY": "k", "B_KEY": "k"}
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            mx = matrix_file(d, [ENTRY])
            samples = canned_samples()
            rm.main(["--matrix", mx, "--json-out", str(out), "--raw-dir", str(pathlib.Path(d) / "raw"), "--quiet"],
                    env=env, run_model=lambda config, lanes, env, log: mf.run(config, lanes=lanes, env=env, post=post),
                    run_samples=samples)
            n = len(sent)
            self.assertGreater(n, 0)
            self.assertFalse(samples.calls[-1][2]["analyze_only"])
            self.assertTrue(samples.calls[-1][2]["raw_dir"].startswith(str(pathlib.Path(d) / "raw")))
            code = rm.main(["--matrix", mx, "--json-out", str(out), "--raw-dir", str(pathlib.Path(d) / "raw"),
                            "--analyze-only", "--quiet"],
                           env=env, run_model=lambda config, lanes, env, log: mf.run(
                               config, lanes=lanes, env=env, post=lambda *a, **k: self.fail("sent in analyze-only")),
                           run_samples=samples)
            rep = json.loads(out.read_text())
        self.assertEqual(len(sent), n)
        self.assertTrue(samples.calls[-1][2]["analyze_only"])
        self.assertEqual(rep["models"][0]["fingerprint"]["source"]["mode"], "analyze_only")
        self.assertEqual(code, rm.EXIT_OK)

    def test_models_filter_and_unknown_id(self):
        two = [ENTRY, dict(ENTRY, id="deeprelay/n", display_name="N")]
        code, rep, _ = self.run_one(baseline(), entries=two, models="deeprelay/n")
        self.assertEqual([m["id"] for m in rep["models"]], ["deeprelay/n"])
        with tempfile.TemporaryDirectory() as d:
            code = rm.main(["--matrix", matrix_file(d, two), "--json-out", str(pathlib.Path(d) / "r.json"),
                            "--models", "deeprelay/zzz", "--quiet"], env={}, run_model=fake_run(lambda m: baseline()),
                           run_samples=canned_samples())
        self.assertEqual(code, rm.EXIT_NOTHING_JUDGED)

    def test_committed_matrix_runs_end_to_end_with_a_fake_post(self):
        """Every committed lane goes through run(); the subject matches."""
        def post_for(lanes):
            roles = {(l["base_url"].rstrip("/") + "/chat/completions", l["model"]): l["role"] for l in lanes}

            def post(url, key, body, **_):
                k = body.get("top_logprobs", 0)
                sig = 40 + len(body["messages"][0]["content"]) % 7
                role = roles[(url, body["model"])]
                if role == "control":
                    return content_shape(["q", "r"], k=k, prompt_tokens=sig * 3)
                if "logprobs" not in body:  # a text lane never asks
                    return no_logprobs(prompt_tokens=sig, content="ab")
                return answer(url, body, k, sig, 0.01 if role == "subject" else 0.0)
            return post

        def answer(url, body, k, sig, spread):
            if body.get("stream"):
                return mf.aggregate_stream(mf.parse_sse([
                    "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "ab"},
                                                        "logprobs": content_shape(["a", "b"], k=k, spread=spread)["choices"][0]["logprobs"]}]}),
                    "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": sig}}),
                    "data: [DONE]"]))
            # Two references a little apart on the chosen token, the subject
            # in between: a k=1 band exists and the subject sits inside it.
            chosen = -0.11 if spread else (-0.10 if "together" in url else -0.12)
            return content_shape(["a", "b"], k=k, prompt_tokens=sig + (3 if "fireworks" in url else 0),
                                 spread=spread, chosen=chosen)
        env = {k: "k" for k in ("DEEPRELAY_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY",
                                "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "ZAI_API_KEY")}
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            # The sampling test judged every reference lane, the vendor's
            # text lanes included (it needs no logprobs).
            samples = canned_samples("pass", judged=("together", "fireworks", "zai", "deepseek", "alibaba"))
            code = rm.main(["--json-out", str(out), "--quiet"], env=env,
                           run_model=lambda config, lanes, env, log: mf.run(config, lanes=lanes, env=env,
                                                                            post=post_for(lanes)),
                           run_samples=samples)
            rep = json.loads(out.read_text())
        self.assertEqual(len(rep["models"]), 5)
        self.assertEqual(len(samples.calls), 5)
        for m in rep["models"]:
            fp = m["fingerprint"]
            self.assertEqual(fp["method"], "model_equality_testing")
            self.assertEqual(fp["samples"]["verdict"], "pass")
            if m["id"] == "deeprelay/glm-5.2":
                # One logprob reference plus the vendor's text lane: the
                # logprob fingerprint cannot judge, the sampling test can,
                # and the vendor lane it judged is a wire "reference".
                self.assertEqual(fp["logprobs_verdict"], "insufficient", fp["reasons"])
                self.assertEqual(fp["verdict"], "pass", fp["reasons"])
                self.assertEqual([l["role"] for l in m["lanes"]], ["subject", "reference", "reference"])
                self.assertEqual(fp["text_references"][0]["lane"], "zai")
                continue
            self.assertEqual(fp["verdict"], "pass", (m["id"], fp["reasons"]))
            self.assertEqual([l for l in m["lanes"] if l["role"] == "text_reference"], [])
            if m["id"].startswith("deeprelay/glm-5.3"):
                self.assertEqual(fp["text_references"][0]["vs"]["deeprelay"]["same_content"], 12)
                self.assertEqual(samples.calls[[c[0] for c in samples.calls].index(m["id"])][1]["request_extra"],
                                 {"reasoning_effort": "low"})
        self.assertEqual(code, rm.EXIT_OK)
        self.assertNotIn("sk-", json.dumps(rep))


class CombinedVerdictTest(unittest.TestCase):
    """The sampling test judges; the logprob fingerprint only vetoes."""

    def test_combine(self):
        self.assertEqual(rm.combine("pass", "pass"), "pass")
        self.assertEqual(rm.combine("inconclusive", "pass"), "pass")
        self.assertEqual(rm.combine("insufficient", "pass"), "pass")
        self.assertEqual(rm.combine("error", "pass"), "pass")
        self.assertEqual(rm.combine("mismatch", "pass"), "mismatch")
        self.assertEqual(rm.combine("pass", "mismatch"), "mismatch")
        self.assertEqual(rm.combine("pass", "inconclusive"), "inconclusive")
        self.assertEqual(rm.combine("pass", "insufficient"), "insufficient")
        self.assertEqual(rm.combine("pass", "error"), "error")
        self.assertEqual(rm.combine("pass", None), "pass")
        self.assertEqual(rm.combine("inconclusive", None), "inconclusive")

    def run_pair(self, responses, samples, env=None):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            argv = ["--matrix", matrix_file(d, [ENTRY]), "--json-out", str(out), "--quiet"]
            if samples == "--no-samples":
                argv.append("--no-samples")
                run_samples = canned_samples()
            else:
                run_samples = canned_samples(samples)
            code = rm.main(argv, env=env or {}, run_model=fake_run(lambda m: responses),
                           run_samples=run_samples)
            return code, json.loads(out.read_text())["models"][0], run_samples

    def subject(self, m):
        return next(l for l in m["lanes"] if l["role"] == "subject")["weights_verdict"]

    def test_logprob_inconclusive_with_samples_pass_is_verified(self):
        # The GLM 5.3 situation: the sibling control cannot be told apart by
        # logprobs, the sampling test tells it apart.
        same_as_refs = lambda i, rep: content_shape(["a", "b", "c"], spread=0.02)
        code, m, _ = self.run_pair(baseline(control=same_as_refs), "pass")
        self.assertEqual(m["fingerprint"]["logprobs_verdict"], "inconclusive")
        self.assertEqual((code, m["fingerprint"]["verdict"], self.subject(m)), (rm.EXIT_OK, "pass", "same_model"))

    def test_samples_inconclusive_is_not_verified_even_if_logprobs_pass(self):
        code, m, _ = self.run_pair(baseline(), "inconclusive")
        self.assertEqual((code, m["fingerprint"]["verdict"], self.subject(m)),
                         (rm.EXIT_OK, "inconclusive", "inconclusive"))

    def test_logprob_mismatch_vetoes(self):
        def far(i, rep):  # a top-5 distribution well outside the band
            d = content_shape(["a", "b", "c"])
            for row in d["choices"][0]["logprobs"]["content"]:
                row["top_logprobs"] = [{"token": row["token"], "logprob": -1.2}] + [
                    {"token": f"other{j}", "logprob": -1.3} for j in range(4)]
            return d
        code, m, _ = self.run_pair(baseline(subject=far), "pass")
        self.assertEqual(m["fingerprint"]["logprobs_verdict"], "mismatch")
        self.assertEqual((code, m["fingerprint"]["verdict"], self.subject(m)),
                         (rm.EXIT_MISMATCH, "mismatch", "mismatch"))
        self.assertEqual(m["fingerprint"]["lane_verdicts"]["subj"]["verdict"], "mismatch")
        self.assertEqual(m["fingerprint"]["samples"]["verdict"], "pass")

    def test_samples_mismatch_is_a_mismatch(self):
        code, m, _ = self.run_pair(baseline(), "mismatch")
        self.assertEqual((code, m["fingerprint"]["verdict"], self.subject(m)),
                         (rm.EXIT_MISMATCH, "mismatch", "mismatch"))

    def test_samples_error_is_not_verified(self):
        code, m, _ = self.run_pair(baseline(), "error")
        self.assertEqual((m["fingerprint"]["verdict"], self.subject(m)), ("error", "error"))

    def test_no_samples_flag_keeps_the_logprob_verdict(self):
        code, m, run_samples = self.run_pair(baseline(), "--no-samples")
        self.assertEqual(run_samples.calls, [])
        self.assertEqual((code, m["fingerprint"]["verdict"], self.subject(m)), (rm.EXIT_OK, "pass", "same_model"))
        self.assertNotIn("samples", m["fingerprint"])
        self.assertNotIn("method", m["fingerprint"])

    def test_settings_and_run_dir_reach_the_sampling_test(self):
        code, m, run_samples = self.run_pair(baseline(), "pass")
        (mid, settings, kw), = run_samples.calls
        self.assertEqual(mid, "deeprelay/m")
        self.assertEqual(settings["request_extra"], {"reasoning_effort": "none"})
        self.assertEqual(settings["per_prompt"], me.PER_PROMPT)
        self.assertEqual(settings["deadline_s"], me.DEADLINE_S)
        self.assertIsNone(settings["max_requests"])
        self.assertNotIn("max_requests", kw)  # the sampling test sizes its own budget

    def test_text_lane_judged_by_sampling_counts_as_a_reference(self):
        entry = dict(ENTRY, lanes=ENTRY["lanes"] + [lane("zai", "reference", "Z_KEY", evidence="text")])
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            argv = ["--matrix", matrix_file(d, [entry]), "--json-out", str(out), "--quiet"]
            rm.main(argv, env={}, run_model=fake_run(lambda m: baseline()),
                    run_samples=canned_samples("pass", judged=("refA", "refB", "zai")))
            roles = {l["name"]: l["role"] for l in json.loads(out.read_text())["models"][0]["lanes"]}
            self.assertEqual(roles["zai"], "reference")
            rm.main(argv, env={}, run_model=fake_run(lambda m: baseline()),
                    run_samples=canned_samples("pass", judged=("refA", "refB")))
            roles = {l["name"]: l["role"] for l in json.loads(out.read_text())["models"][0]["lanes"]}
            self.assertEqual(roles["zai"], "text_reference")
            rm.main(argv + ["--no-samples"], env={}, run_model=fake_run(lambda m: baseline()),
                    run_samples=canned_samples("pass"))
            roles = {l["name"]: l["role"] for l in json.loads(out.read_text())["models"][0]["lanes"]}
            self.assertEqual(roles["zai"], "text_reference")


class PublishTest(unittest.TestCase):
    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(self, script):
        calls = []

        def open_(req, timeout=None):
            calls.append(req)
            outcome = script.pop(0)
            if isinstance(outcome, int):
                raise urllib.error.HTTPError(req.full_url, outcome, "x", {}, io.BytesIO(b'{"error":"bad"}'))
            if isinstance(outcome, Exception):
                raise outcome
            return self.Resp(outcome)
        return open_, calls

    def report_file(self, d):
        f = pathlib.Path(d) / "r.json"
        f.write_text(json.dumps({"schema_version": 1, "models": [1]}))
        return str(f)

    def test_success_prints_run_id_and_stored(self):
        open_, calls = self.opener([b'{"run_id":"123","run_attempt":1,"stored":true}'])
        out = io.StringIO()
        real = sys.stdout
        sys.stdout = out
        try:
            with tempfile.TemporaryDirectory() as d:
                code = publish.main([self.report_file(d), "--url", "https://x/ingest", "--token-env", "T"],
                                    env={"T": "tok"}, opener=open_, sleep=lambda s: None)
        finally:
            sys.stdout = real
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), {"run_id": "123", "stored": True})
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer tok")
        self.assertEqual(calls[0].get_header("Content-type"), "application/json")

    def test_5xx_and_transport_are_retried_three_times(self):
        open_, calls = self.opener([503, ConnectionError("down"), b'{"run_id":"1","stored":true}'])
        slept = []
        with tempfile.TemporaryDirectory() as d:
            code = publish.main([self.report_file(d), "--url", "https://x", "--token-env", "T"],
                                env={"T": "tok"}, opener=open_, sleep=slept.append)
        self.assertEqual((code, len(calls)), (0, 3))
        self.assertEqual(slept, [publish.BACKOFF_S, 2 * publish.BACKOFF_S])
        open_, calls = self.opener([500, 502, 503])
        with tempfile.TemporaryDirectory() as d:
            code = publish.main([self.report_file(d), "--url", "https://x", "--token-env", "T"],
                                env={"T": "tok"}, opener=open_, sleep=lambda s: None)
        self.assertEqual((code, len(calls)), (1, 3))

    def test_4xx_is_final_and_non_zero(self):
        open_, calls = self.opener([400])
        err = io.StringIO()
        real = sys.stderr
        sys.stderr = err
        try:
            with tempfile.TemporaryDirectory() as d:
                code = publish.main([self.report_file(d), "--url", "https://x", "--token-env", "T"],
                                    env={"T": "tok"}, opener=open_, sleep=lambda s: None)
        finally:
            sys.stderr = real
        self.assertEqual((code, len(calls)), (1, 1))
        self.assertIn('{"error":"bad"}', err.getvalue())

    def test_missing_token_is_an_error_without_a_request(self):
        open_, calls = self.opener([])
        with tempfile.TemporaryDirectory() as d:
            code = publish.main([self.report_file(d), "--url", "https://x", "--token-env", "T"],
                                env={}, opener=open_, sleep=lambda s: None)
        self.assertEqual((code, calls), (2, []))


if __name__ == "__main__":
    unittest.main()
