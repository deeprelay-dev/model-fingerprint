# model-fingerprint

Are the open-weight models served at `api.deeprelay.ai` the real weights, at
comparable precision, and not a quantized or substituted build?

This repository is the harness that answers that question, continuously, in
public. It sends twelve fixed prompts at temperature 0 with log-probabilities
to the deeprelay endpoint and to two independently operated deployments of
the same weights, and judges the deeprelay lane against the band those two
references define. Negative controls (deliberately different models) prove the
measurement can tell models apart at all. Every number behind a verdict is in
the report; every run is traceable to the GitHub Actions job that produced
it.

It is plain Python 3.10+ with no dependencies, so you can run it yourself
against `api.deeprelay.ai` with your own deeprelay key and your own keys for
the reference providers, and get the same verdicts we publish.

The method, its calibration and its known limits are in [METHOD.md](METHOD.md).

## What it proves

For each model in [`matrix.json`](matrix.json):

- **`same_model`**: the deeprelay lane's top-k logprob distributions over the
  content tokens are no further from the two reference deployments than
  those references are from each other (within a 1.5x tolerance, widened by
  at most 2x the band for measured replicate noise, under an absolute
  ceiling), and its tokenizer-plus-template signature agrees.
- **`mismatch`**: the signature differs, or the distribution distance
  (top-k width 2 or more) is outside that limit while a control shows the
  metric discriminates. Top-1 evidence (a lane served at `top_logprobs` 1)
  can confirm a match but never fails a lane on its own: over the limit
  it reads `inconclusive`, because at that width noise and substitution
  cannot be told apart. A model that had
  to think before answering (thinking cannot be turned off) is never
  failed on distance alone, because each lane's answer conditions on its
  own reasoning; it can only be `inconclusive`.
- **`inconclusive`**: the lane could not be judged either way. The common
  causes are a control that did not separate (so the metric has not been
  shown to discriminate for this model), a reference band over its ceiling,
  a lane too noisy to judge, or too few prompts sharing a greedy prefix.
- **`insufficient`**: something was not measured: a key missing, a lane
  returning fewer than 10 of 12 prompts, fewer than two independent
  references with top-k logprobs.

Models the relay cannot serve at temperature 0 are not listed, because the
method compares greedy paths and a sampled lane cannot be compared to a
greedy one; Kimi K3 is the current example (the relay pins its temperature
to 1.0 for that upstream).

A model is marked **verified** on deeprelay's public surfaces only when the
run verdict is `pass` (subject `same_model` AND every control separated AND
every metric the verdict relied on was separated by a control). Anything
else renders nothing. A run that measures nothing exits non-zero.

## Run it yourself

```bash
git clone https://github.com/deeprelay-dev/model-fingerprint
cd model-fingerprint

export DEEPRELAY_API_KEY=...      # your deeprelay key (the subject lane)
export TOGETHER_API_KEY=...       # reference and control lanes
export FIREWORKS_API_KEY=...
export DASHSCOPE_API_KEY=...      # Qwen 3.8 Flash reference (Alibaba)

# the whole matrix (about 480 tiny requests, under ten cents)
python3 run_matrix.py --json-out report.json --raw-dir raw --summary-out summary.md

# one model, with the six-section human report
python3 model_fingerprint.py --model deeprelay/deepseek-v4.1-flash --json-out fp.json

# re-analyze the last saved run of a model without sending anything
python3 model_fingerprint.py --model deeprelay/deepseek-v4.1-flash --analyze-only
```

Keys are read from the environment variables the matrix names. Nothing in
the matrix, the report, the logs or the raw responses carries a key. A lane
whose key is not set is skipped loudly and the model reads `insufficient`;
it is never silently dropped.

Exit codes of `run_matrix.py`: `0` when at least one model was judged and no
model is a mismatch, `2` on any mismatch, `1` when nothing could be judged.

### Reading a verdict

Each model's report has six sections: within-lane determinism (replicate vs
replicate), cross-lane distances, the per-prompt emitted content, the
tokenizer/template signature, the two-sample summary with the band and each
lane's distance and limit, and the negative controls. The last line is the
verdict with its reasons. A compact table goes to `summary.md` (and to the
job summary in CI):

| model | verdict | subject | k | distance | limit | band | signature | why |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | **pass** | same_model | 1 | 0.0100 | 0.0243 | 0.0141 | 12/12 | lp_delta 0.0100 <= limit 0.0243 ... |

`k` is the top-k width the lane was judged at. The subject lane asks the
relay for `top_logprobs: 5`; where the relay caps a model lower it answers
a 400, and the harness steps the same probe down to 1 (then to no logprobs)
and records the width actually served (`topk_served`). A lane judged at
k=1 is compared on the chosen token's log-probability instead of a
distribution. `distance` is the lane's
mean distance to the references, `limit` what it had to stay under, `band`
the references' mutual distance at that k.

## Cost

A full matrix run is about 480 requests of a few tokens each (two
replicates per prompt on subject and reference lanes, one on controls;
thinking-only models use `max_tokens` 2000). On list prices that is on the
order of ten US cents per run. The scheduled workflow runs four times a day.

## Repository

| file | purpose |
|---|---|
| `model_fingerprint.py` | the harness: collection, metrics, verdicts, human report; `run()` library entry and CLI |
| `run_matrix.py` | runs every model in `matrix.json` and writes one report in the deeprelay benchmark schema |
| `publish.py` | POSTs a report to a deeprelay ingest endpoint |
| `matrix.json` | the models, their lanes (subject, references, controls), per-model request settings |
| `test_model_fingerprint.py`, `test_run_matrix.py` | hermetic unit tests (`python3 -m unittest discover -s . -p 'test_*.py'`) |
| `.github/workflows/verify.yml` | scheduled run plus publish (schedule and dispatch only, never on pull requests) |
| `.github/workflows/ci.yml` | unit tests on pull requests and pushes, no secrets |
| `METHOD.md` | the method, calibration numbers, what it can and cannot exclude, change log of the method |

Report consumers: `GET https://api.deeprelay.ai/v1/benchmarks/latest` and the
`deeprelay benchmarks latest` CLI command decode the same document this
harness writes (`schema_version` 1).

## Contributing

Pull requests run the unit tests only. Adding a model means adding a matrix
entry with the deeprelay id, the weights, the subject lane, two independent
reference lanes serving the same weights (not the deeprelay lane's own
upstream; the harness detects and rejects that), and at least one control
the method can separate. Keep `METHOD.md` honest about what each model's
controls do and do not exclude.

## License

Apache-2.0, Copyright 2026 Deeprelay, Inc.
