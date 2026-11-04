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
    def run_one(self, responses, entries=None, models=None, env=None):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            summ = pathlib.Path(d) / "summary.md"
            argv = ["--matrix", matrix_file(d, entries or [ENTRY]), "--json-out", str(out),
                    "--summary-out", str(summ), "--quiet"]
            if models:
                argv += ["--models", models]
            code = rm.main(argv, env=env or {}, run_model=fake_run(lambda m: responses))
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
        self.assertEqual(fp["requests"], 7)
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
        self.assertEqual(rep["summary"]["requests"], 7)
        self.assertEqual(rep["summary"]["verdicts"], {"pass": 1})
        self.assertEqual(rep["summary"]["errors"], {"rate_limited": 1})
        self.assertIn("| M | **pass** | same_model |", md)

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
        code, rep, _ = self.run_one(r, entries=[dict(ENTRY, lanes=lanes)])
        self.assertEqual(code, rm.EXIT_OK)
        # The lane matched; the RUN is inconclusive because no control
        # separated. The server marks a model verified only when both the
        # fingerprint verdict is pass and the subject lane is same_model.
        self.assertEqual(rep["models"][0]["fingerprint"]["verdict"], "inconclusive")
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "same_model")

    def test_nothing_judged_exits_1(self):
        r = {k: v for k, v in baseline().items() if k[0] != "subj"}
        code, rep, _ = self.run_one(r)
        self.assertEqual(code, rm.EXIT_NOTHING_JUDGED)
        self.assertEqual(rep["models"][0]["lanes"][0]["weights_verdict"], "insufficient")
        self.assertEqual(rep["models"][0]["fingerprint"]["verdict"], "insufficient")

    def test_error_model_still_produces_a_decodable_row(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            code = rm.main(["--matrix", matrix_file(d, [ENTRY]), "--json-out", str(out), "--quiet"],
                           env={}, run_model=fake_run(lambda m: None))
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
                               post=lambda u, k, b, **_: content_shape(["a"], prompt_tokens=99 if "ctrl" in u else 40)))
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
            rm.main(["--matrix", mx, "--json-out", str(out), "--raw-dir", str(pathlib.Path(d) / "raw"), "--quiet"],
                    env=env, run_model=lambda config, lanes, env, log: mf.run(config, lanes=lanes, env=env, post=post))
            n = len(sent)
            self.assertGreater(n, 0)
            code = rm.main(["--matrix", mx, "--json-out", str(out), "--raw-dir", str(pathlib.Path(d) / "raw"),
                            "--analyze-only", "--quiet"],
                           env=env, run_model=lambda config, lanes, env, log: mf.run(
                               config, lanes=lanes, env=env, post=lambda *a, **k: self.fail("sent in analyze-only")))
            rep = json.loads(out.read_text())
        self.assertEqual(len(sent), n)
        self.assertEqual(rep["models"][0]["fingerprint"]["source"]["mode"], "analyze_only")
        self.assertEqual(code, rm.EXIT_OK)

    def test_models_filter_and_unknown_id(self):
        two = [ENTRY, dict(ENTRY, id="deeprelay/n", display_name="N")]
        code, rep, _ = self.run_one(baseline(), entries=two, models="deeprelay/n")
        self.assertEqual([m["id"] for m in rep["models"]], ["deeprelay/n"])
        with tempfile.TemporaryDirectory() as d:
            code = rm.main(["--matrix", matrix_file(d, two), "--json-out", str(pathlib.Path(d) / "r.json"),
                            "--models", "deeprelay/zzz", "--quiet"], env={}, run_model=fake_run(lambda m: baseline()))
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
            code = rm.main(["--json-out", str(out), "--quiet"], env=env,
                           run_model=lambda config, lanes, env, log: mf.run(config, lanes=lanes, env=env,
                                                                            post=post_for(lanes)))
            rep = json.loads(out.read_text())
        self.assertEqual(len(rep["models"]), 5)
        for m in rep["models"]:
            fp = m["fingerprint"]
            if m["id"] == "deeprelay/glm-5.2":
                # One logprob reference plus the vendor's text lane: reported,
                # never judged, and the text lane is not a wire "reference".
                self.assertEqual(fp["verdict"], "insufficient", fp["reasons"])
                self.assertEqual([l["role"] for l in m["lanes"]], ["subject", "reference", "text_reference"])
                self.assertEqual(fp["text_references"][0]["lane"], "zai")
                continue
            self.assertEqual(fp["verdict"], "pass", (m["id"], fp["reasons"]))
            text = [l for l in m["lanes"] if l["role"] == "text_reference"]
            if m["id"].startswith("deeprelay/glm-5.3"):
                self.assertEqual([l["name"] for l in text], ["zai"])
                self.assertEqual(fp["text_references"][0]["vs"]["deeprelay"]["same_content"], 12)
        self.assertEqual(code, rm.EXIT_OK)
        self.assertNotIn("sk-", json.dumps(rep))


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
