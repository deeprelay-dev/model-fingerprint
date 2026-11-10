# model-fingerprint

Are the open-weight models served at `api.deeprelay.ai` the real weights, at
comparable precision, and not a quantized or substituted build?

This repository is the harness that answers that question, in public, with
a published, peer-reviewed test rather than a method of deeprelay's own.

The judge is **Model Equality Testing** (Gao, Liang and Guestrin, ICLR
2025, [arXiv:2410.20247](https://arxiv.org/abs/2410.20247)), run through
the authors' package [`model-equality-testing`](https://github.com/i-gao/model-equality-testing)
exactly as published: sixteen prompts are sampled at temperature 1 from
the deeprelay endpoint, from the model vendor's own API and from
independently operated deployments of the same weights, and the package's
two-sample test (maximum mean discrepancy with the Hamming kernel, a
permutation p-value) says whether the deeprelay lane's completions are
drawn from the same distribution as the references'. Because it needs
samples, not log-probabilities, every vendor API is a full reference.
Negative controls (deliberately different models, the nearest sibling
checkpoint where one exists) prove the test can tell models apart on that
day's samples. The top-k **logprob fingerprint** this harness started with
still runs on every host that returns logprobs and is reported beside it
as corroborating evidence; it can veto a verdict, not grant one.

Every number behind a verdict is in the report; every run is traceable to
the GitHub Actions job that produced it. deeprelay runs it once a day and
publishes the report; the verified mark on deeprelay.ai reflects the latest
daily run. You can run the same thing yourself against `api.deeprelay.ai`
with your own deeprelay key and your own vendor keys (DeepSeek, Z.ai,
Alibaba, Together, Fireworks) and get the same verdicts.

The method, its calibration and its known limits are in [METHOD.md](METHOD.md).

## What it proves

For each model in [`matrix.json`](matrix.json), the sampling test judges
the deeprelay lane:

- **`same_model`**: the deeprelay lane's sampled completions match its
  nearest reference deployment (the package's test does not reject
  equality at p >= 0.05, or the MMD distance is inside the band the
  reference deployments span among themselves), and the lane is closer to
  every reference than to any control.
- **`mismatch`**: the lane is outside the band against every reference
  and at least as far from the nearest reference as it is from the nearest
  control: it looks like a different model, not like the same weights on
  another serving stack.
- **`inconclusive`**: the lane could not be judged either way: a control
  did not separate (so the test has not been shown to discriminate on that
  day's samples), or the lane is outside the band but still much closer
  to the references than to any control.
- **`insufficient`**: something was not measured: a key missing, a lane
  returning fewer than 80% of its planned samples, fewer than two
  reference deployments.

Why a band rather than the bare p-value: two deployments of the same
weights on different serving stacks (a different quantisation, sampler,
chat template or thinking budget) do not produce identical distributions,
and the test is sensitive enough to say so; on 2026-11-10 the three
reference deployments of GLM 5.3 rejected each other at p < 0.05 while
GLM 5.3 and GLM 5.3 Flash sat three to four times further apart (see
METHOD.md). So, as in the logprob fingerprint, the references' mutual
distances define what "same weights, different host" looks like, and the
subject is judged against that, with the controls proving the gap to a
different model is wide.

The **logprob fingerprint** (twelve prompts at temperature 0 with
`top_logprobs`, judged against the band two logprob references define)
runs alongside on every host that returns log-probabilities and is printed
in full in the report, with its own verdict under `logprobs_verdict`. It
corroborates: a logprob `mismatch` makes the model a mismatch whatever the
sampling test says; a logprob `inconclusive` or `insufficient` changes
nothing (GLM 5.3's sibling control cannot be told apart by logprobs,
which is exactly why the sampling test is the judge). `--no-samples`
runs the logprob fingerprint alone, as the harness did before 2026.11.10.

Kimi K3 is not listed yet. Moonshot's own API accepts only temperature 1
for `kimi-k3`, which suits the sampling test, but serves at 3 requests a
minute, so a full sample takes over an hour and needs its own schedule.

A model is marked **verified** on deeprelay's public surfaces only when the
run verdict is `pass` (sampling test `pass`, subject `same_model`, at
least one gating control separated, no logprob mismatch). Anything else
renders nothing. A run that measures nothing exits non-zero.

## Run it yourself

```bash
git clone https://github.com/deeprelay-dev/model-fingerprint
cd model-fingerprint
# the published test's package (torch is its heaviest dependency; the CPU build is enough)
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
python3 -m pip install -r requirements.txt

export DEEPRELAY_API_KEY=...      # your deeprelay key (the subject lane)
export DEEPSEEK_API_KEY=...       # DeepSeek's own API: V4.1 Flash reference
export DASHSCOPE_API_KEY=...      # Alibaba's own API: Qwen 3.8 Flash reference
export ZAI_API_KEY=...            # Z.ai's own API: GLM 5.2, 5.3, 5.3 Flash reference
export TOGETHER_API_KEY=...       # independent references and controls
export FIREWORKS_API_KEY=...

# the whole matrix (about 480 logprob probes plus about 1,200 sampled completions per model)
python3 run_matrix.py --json-out report.json --raw-dir raw --summary-out summary.md

# one model, both tests
python3 run_matrix.py --models deeprelay/glm-5.3 --json-out report.json --raw-dir raw

# the logprob fingerprint alone (no package needed), with its six-section human report
python3 model_fingerprint.py --model deeprelay/deepseek-v4.1-flash --json-out fp.json

# re-judge the last saved run without sending anything
python3 run_matrix.py --json-out report.json --raw-dir raw --analyze-only
```

Keys are read from the environment variables the matrix names. Nothing in
the matrix, the report, the logs or the raw responses carries a key. A lane
whose key is not set is skipped loudly and the model reads `insufficient`;
it is never silently dropped.

What each key buys you:

| key | lanes | what it adds |
|---|---|---|
| `DEEPRELAY_API_KEY` | the subject lane of every model | required; nothing is judged without it |
| `DEEPSEEK_API_KEY` | DeepSeek V4.1 Flash reference (`deepseek-flash`, top-5 logprobs) | the vendor's own deployment in the band |
| `DASHSCOPE_API_KEY` | Qwen 3.8 Flash reference (Alibaba, top-5 logprobs) | the vendor's own deployment in the band |
| `ZAI_API_KEY` | GLM 5.2, 5.3, 5.3 Flash reference (Z.ai) | the vendor's own deployment in the band; Z.ai offers no logprobs, so it takes part in the sampling test only |
| `TOGETHER_API_KEY` | references and controls for every model | the second independent deployment |
| `FIREWORKS_API_KEY` | references and controls | the second or third independent deployment; MiniMax M3 control |

Any subset works. A model needs its subject lane, two reference
deployments and one control to be judged; everything else only adds
evidence.

Exit codes of `run_matrix.py`: `0` when at least one model was judged and no
model is a mismatch, `2` on any mismatch, `1` when nothing could be judged.

### Reading a verdict

Each model prints the logprob fingerprint's six-section report (within-lane
determinism, cross-lane distances, emitted content, the tokenizer/template
signature, the two-sample summary, the negative controls) and then the
SAMPLES block: every lane's sample count, the band the reference
deployments span, each control's distance and p-value against the subject,
and the subject's distance and p-value against every reference. A compact
table goes to `summary.md` (and to the job summary in CI):

| model | verdict | samples (MET) | mmd | band | p | control | logprobs | distance | limit | why |
|---|---|---|---|---|---|---|---|---|---|---|
| GLM 5.3 | **pass** | pass | 0.0100 | 0.0222 | 0.210 | ctrl_glm53flash yes, ctrl_minimax_m3 yes, ctrl_v41flash yes | inconclusive | 0.0310 | 0.0460 | matches the nearest reference (nearest together: mmd 0.0100, p=0.210; band 0.0222) ... |

`mmd` and `p` are the subject's MMD distance and permutation p-value
against its nearest reference deployment; `band` the largest MMD between
two reference deployments; `control` whether each control separated.
`logprobs`, `distance` and `limit` are the logprob fingerprint's verdict
and numbers, kept for corroboration.

A model with a text lane also prints section 4b, the lane's agreement with
every other lane, and a note in the verdict block such as "text lane zai:
deeprelay gave identical content on 12/12 prompts; the controls at most
2/12; text lanes inform, they do not judge".

`k` is the top-k width the lane was judged at. The subject lane asks the
relay for `top_logprobs: 5`; where the relay caps a model lower it answers
a 400, and the harness steps the same probe down to 1 (then to no logprobs)
and records the width actually served (`topk_served`). A lane judged at
k=1 is compared on the chosen token's log-probability instead of a
distribution. `distance` is the lane's
mean distance to the references, `limit` what it had to stay under, `band`
the references' mutual distance at that k.

## Cost

A full matrix run is about 480 logprob probes of a few tokens each plus,
per model, sixteen prompts times fifteen samples per lane at up to 1,200
tokens (thinking models spend most of that on reasoning): roughly 1,500
sampled completions per model, 7,000 per run. On list prices that is on
the order of one to two US dollars per run. The scheduled workflow runs
once a day.

## Repository

| file | purpose |
|---|---|
| `model_equality.py` | the sampling test: collects samples at temperature 1 from every lane and judges them with the published `model-equality-testing` package (Gao, Liang, Guestrin, ICLR 2025) |
| `model_fingerprint.py` | the logprob fingerprint: collection, metrics, verdicts, human report; `run()` library entry and CLI |
| `run_matrix.py` | runs both tests on every model in `matrix.json`, combines the verdicts and writes one report in the deeprelay benchmark schema |
| `requirements.txt` | the published package and its (partly undeclared) dependencies |
| `publish.py` | POSTs a report to a deeprelay ingest endpoint |
| `matrix.json` | the models, their lanes (subject, references, controls), per-model request settings |
| `test_model_equality.py`, `test_model_fingerprint.py`, `test_run_matrix.py` | hermetic unit tests (`python3 -m unittest discover -s . -p 'test_*.py'`) |
| `.github/workflows/verify.yml` | the daily run plus publish (schedule and dispatch only, never on pull requests) |
| `.github/workflows/ci.yml` | unit tests on pull requests and pushes, no secrets |
| `METHOD.md` | the method, calibration numbers, what it can and cannot exclude, change log of the method |

Report consumers: `GET https://api.deeprelay.ai/v1/benchmarks/latest` and the
`deeprelay benchmarks latest` CLI command decode the same document this
harness writes (`schema_version` 1).

## Contributing

Pull requests run the unit tests only. Adding a model means adding a matrix
entry with the deeprelay id, the weights, the subject lane, two independent
reference deployments of the same weights (the vendor's own API first;
never the deeprelay lane's own upstream, which the logprob fingerprint
detects and rejects; a vendor without logprobs is marked `evidence:
"text"` and takes part in the sampling test only), and at least one
control the test separates, the nearest sibling checkpoint where one is
served. A thinking-only model needs a `samples.request_extra` that keeps
its reasoning short. Keep `METHOD.md` honest about what each model's
controls do and do not exclude.

## License

Apache-2.0, Copyright 2026 Deeprelay, Inc.
