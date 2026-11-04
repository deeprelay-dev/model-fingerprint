# model-fingerprint

Are the open-weight models served at `api.deeprelay.ai` the real weights, at
comparable precision, and not a quantized or substituted build?

This repository is the harness that answers that question, in public. It
sends twelve fixed prompts at temperature 0 with log-probabilities to the
deeprelay endpoint, to the model vendor's own API, and to independently
operated deployments of the same weights, and judges the deeprelay lane
against the band two logprob references define. Negative controls
(deliberately different models) prove the measurement can tell models apart
at all. Every number behind a verdict is in the report; every run is
traceable to the GitHub Actions job that produced it.

deeprelay runs it once a day and publishes the report; the verified mark on
deeprelay.ai reflects the latest daily run. It is plain Python 3.10+ with no
dependencies, so you can run the same thing yourself against
`api.deeprelay.ai` with your own deeprelay key and your own vendor keys
(DeepSeek, Z.ai, Alibaba, Together, Fireworks) and get the same verdicts.

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

A lane whose API offers no log-probabilities, which today means Z.ai for
the GLM models, is a **text lane**: it is probed exactly like the others and
its greedy answers and prompt_tokens shape are compared with every lane and
printed in the report, but it never defines a band and never changes a
verdict. With a Z.ai key you can put Z.ai's own answers next to the
relay's; the verdict still rests on the logprob references. Z.ai is not
deterministic at temperature 0 (it reproduces its own answers on 6 to 9 of
12 prompts), so expect the content agreement to be partial and read the
signature column, which is exact.

Kimi K3 is not listed. Moonshot's own API refuses `logprobs` and accepts
only temperature 1 for `kimi-k3`, and the relay accepts only temperature 1
for it as well, so there is no greedy path to compare on either side.

A model is marked **verified** on deeprelay's public surfaces only when the
run verdict is `pass` (subject `same_model` AND every control separated AND
every metric the verdict relied on was separated by a control). Anything
else renders nothing. A run that measures nothing exits non-zero.

## Run it yourself

```bash
git clone https://github.com/deeprelay-dev/model-fingerprint
cd model-fingerprint

export DEEPRELAY_API_KEY=...      # your deeprelay key (the subject lane)
export DEEPSEEK_API_KEY=...       # DeepSeek's own API: V4.1 Flash reference
export DASHSCOPE_API_KEY=...      # Alibaba's own API: Qwen 3.8 Flash reference
export ZAI_API_KEY=...            # Z.ai's own API: GLM text lanes (no logprobs offered)
export TOGETHER_API_KEY=...       # independent references and controls
export FIREWORKS_API_KEY=...

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

What each key buys you:

| key | lanes | what it adds |
|---|---|---|
| `DEEPRELAY_API_KEY` | the subject lane of every model | required; nothing is judged without it |
| `DEEPSEEK_API_KEY` | DeepSeek V4.1 Flash reference (`deepseek-flash`, top-5 logprobs) | the vendor's own deployment in the band |
| `DASHSCOPE_API_KEY` | Qwen 3.8 Flash reference (Alibaba, top-5 logprobs) | the vendor's own deployment in the band |
| `ZAI_API_KEY` | GLM 5.2, 5.3, 5.3 Flash text lanes | the vendor's own answers, compared and reported; no logprobs offered |
| `TOGETHER_API_KEY` | references and controls for every model | the second independent logprob deployment |
| `FIREWORKS_API_KEY` | references and controls | the second or third independent logprob deployment; MiniMax M3 control |

Any subset works. A model needs its subject lane plus two logprob
references to be judged; everything else only adds evidence.

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

A full matrix run is about 480 requests of a few tokens each (two
replicates per prompt on subject and reference lanes, one on controls;
thinking-only models use `max_tokens` 2000). On list prices that is on the
order of ten US cents per run. The scheduled workflow runs once a day.

## Repository

| file | purpose |
|---|---|
| `model_fingerprint.py` | the harness: collection, metrics, verdicts, human report; `run()` library entry and CLI |
| `run_matrix.py` | runs every model in `matrix.json` and writes one report in the deeprelay benchmark schema |
| `publish.py` | POSTs a report to a deeprelay ingest endpoint |
| `matrix.json` | the models, their lanes (subject, references, controls), per-model request settings |
| `test_model_fingerprint.py`, `test_run_matrix.py` | hermetic unit tests (`python3 -m unittest discover -s . -p 'test_*.py'`) |
| `.github/workflows/verify.yml` | the daily run plus publish (schedule and dispatch only, never on pull requests) |
| `.github/workflows/ci.yml` | unit tests on pull requests and pushes, no secrets |
| `METHOD.md` | the method, calibration numbers, what it can and cannot exclude, change log of the method |

Report consumers: `GET https://api.deeprelay.ai/v1/benchmarks/latest` and the
`deeprelay benchmarks latest` CLI command decode the same document this
harness writes (`schema_version` 1).

## Contributing

Pull requests run the unit tests only. Adding a model means adding a matrix
entry with the deeprelay id, the weights, the subject lane, two independent
reference lanes with logprobs serving the same weights (the vendor's own API
first, when it returns logprobs; never the deeprelay lane's own upstream,
which the harness detects and rejects), a text lane for a vendor that offers
no logprobs, and at least one control the method can separate. Keep `METHOD.md` honest about what each model's
controls do and do not exclude.

## License

Apache-2.0, Copyright 2026 Deeprelay, Inc.
