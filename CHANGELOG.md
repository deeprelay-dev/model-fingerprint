# Changelog

The harness version is the date of the last change to prompts, thresholds
or the measurement loop, and is written into every report as
`harness_version`. Reports from different harness versions are not
comparable.

## 2026.11.10

- The judge is now Model Equality Testing (Gao, Liang, Guestrin, ICLR
  2025), run through the authors' published package
  `model-equality-testing` 0.0.2: every lane is sampled at temperature 1
  on sixteen new prompts, fifteen times each, and the package's
  `mmd_hamming` statistic and permutation p-value decide. The verdict is
  calibrated the way the logprob fingerprint is: the reference
  deployments' mutual MMD is the band, the subject must match its nearest
  reference (p >= 0.05 or inside the band) and be closer to every
  reference than to any control, and a gating control must separate with
  p < 0.05 (`model_equality.py`; METHOD.md, "The judge since 2026.11.10").
- The logprob fingerprint corroborates: it still runs on every host with
  logprobs, its report and verdict are published as `logprobs_verdict`,
  a logprob `mismatch` vetoes, an `inconclusive` or `insufficient`
  changes nothing. `run_matrix.py --no-samples` restores the old
  behaviour.
- Vendor APIs without logprobs (Z.ai) are full references in the
  sampling test; a text lane the sampling test judged is reported as a
  `reference` on the wire.
- Measured before shipping: GLM 5.3 and GLM 5.3 Flash, inconclusive on
  every daily run since 2026-11-04 because logprobs could not tell the
  siblings apart, both read `same_model` with all three sibling controls
  rejected at p = 0.000 (numbers in METHOD.md, "Changes on 2026-11-10").
- `requirements.txt`; CI on Python 3.11 to 3.13; matrix `defaults.samples`
  and per-model `samples` blocks; report `settings.samples_*`.

## 2026.11.06

- Measurement loop: a 429, 5xx or transport error is retried four times
  (was two) with growing backoff, and every probe that still failed on a
  retryable error is sent once more after the whole sweep. On 2026-11-06
  Fireworks answered "service overloaded" on 3 of 12 prompts of the only
  gating control for DeepSeek V4.1 Flash, two retries over 6 s did not
  outlast it, the control was dropped at 9/12 and the model read
  `inconclusive` with its subject lane at `same_model`. A re-probed record
  replaces the failed one and says so (`reprobed`, `first_error`); the
  report's `source.reprobed` and the summary's SOURCE line count them.
- Matrix: a second gating negative control on a different host for every
  model that had only one, so one host's outage cannot leave a model with
  no gating control: Qwen 3.8 Flash on Together for DeepSeek V4.1 Flash;
  DeepSeek V4.1 Flash 1017 on Together for GLM 5.2, GLM 5.3, GLM 5.3
  Flash and Qwen 3.8 Flash (for Qwen a third control, the first that
  separates on JS distance with margin). Measured before shipping (see
  METHOD.md, "Changes on 2026-11-06"). No change to prompts, thresholds
  or the metrics.

## 2026.11.04.1

- Text lanes (`"evidence": "text"`): a reference deployment whose API offers
  no logprobs is probed at temperature 0, its greedy content and signature
  shape are compared with every lane and reported (`text_references`,
  report section 4b, a note per subject lane), and it never judges: not a
  band, not one of the two logprob references, not a verdict change. On
  the wire such a lane carries role `text_reference`, so the deeprelay API
  does not count it among the deployments a verified model matched.
- Matrix: DeepSeek's own API (`deepseek-flash`, top-5 logprobs) is a
  reference for DeepSeek V4.1 Flash. Z.ai (no logprobs) is a text lane for
  GLM 5.2, GLM 5.3 and GLM 5.3 Flash. GLM 5.2 is back, with Together as its
  one logprob reference, and reads `insufficient` until a second exists.
- The schedule is daily. No change to prompts, thresholds or the metrics;
  the version bump marks the new report fields.

## 2026.11.04

- Matrix: GLM 5.2 is out until a second independent reference exists.
  Together is the only independent deployment that returns logprobs for
  it, and the candidate second reference measured on 2026-11-03 was the
  relay's own upstream (see "Reference independence" in METHOD.md). A
  model the harness cannot judge is not run on the schedule.
- References are the model vendors' own APIs where those return token
  logprobs (Alibaba for Qwen 3.8 Flash), plus Together and Fireworks.
  Z.ai's API reference lists no logprobs, so GLM has no vendor lane yet.
- No change to prompts, thresholds or the measurement loop; reports from
  2026.11.03 and 2026.11.04 differ only in the matrix.

## 2026.11.03

First public release. Ported from the private deeprelay livecheck tool and
fixed for the behaviour measured on 2026-11-03:

- Content-token alignment: each response's logprob stream is sliced to the
  window that spells `message.content` before anything is compared.
  Reasoning tokens (which some providers include in the stream and others
  do not) and trailing stop tokens are never compared. A prompt whose
  content is empty counts as not aligned, never as a distance of 1.0.
- Per-model `request_extra` applied to every lane of a model (so all lanes
  condition on the same thing, for example `reasoning_effort: none`), with
  an optional per-lane `request_extra` merged on top (a null value removes
  a key), and per-model `max_tokens`.
- Streaming lanes (`"stream": true`): SSE deltas are folded back into the
  non-stream response shape, so a provider that serves a model stream-only
  can be a reference.
- The tokenizer/template signature is compared as a relative shape (each
  count minus the lane's own count on the anchor prompt), so a provider
  that prepends a fixed system prompt still agrees. Absolute counts and the
  per-lane offset are kept in the report. The control signature-separation
  threshold is re-derived for relative shapes: 2 of 12 prompts.
- Width-aware distances: every comparison truncates both sides to the
  common k, a lane is judged at its own width (capped at 5) against a band
  computed at that width, and `bands_by_k` carries the detail.
- Reference independence: a reference the subject reproduces byte-for-byte
  within that reference's own replicate noise (capped at 0.004) is
  recognised as the subject's own upstream and does not count as a
  reference.
- Advisory controls: a control marked `advisory` measures resolution (a
  sibling checkpoint at the edge of what the metric resolves), is reported
  every run, and does not gate the verdict.
- Thinking-on lanes cannot be failed on distance alone: when the subject or
  a reference thought before answering on at least half the prompts, a
  distance over the limit reads `inconclusive` (the content conditions on
  each lane's own reasoning), never `mismatch`. The signature still fails
  such a lane. Per-lane `thinking_prompts` is reported.
- Retries: 429, 5xx and transport errors are retried twice with backoff.
- Subject width step-down: the subject asks for `top_logprobs: 5`; a 400
  that rejects `top_logprobs` steps the same probe down to 1, then to no
  logprobs. The width served is reported (`topk_served`).
- `--relay-base` retargets the configured subject lane instead of adding
  a second verdict lane; a lanes file without a subject still gets the
  relay lane appended.
- Models the relay cannot serve at temperature 0 are not listed (Kimi K3).
- Top-1 evidence never fails a lane: at width 1 a chosen-token delta over
  the limit reads `inconclusive` ("a wider top_logprobs is needed"), not
  `mismatch`. A mismatch needs the signature or a distribution distance at
  width 2 or more. Prompted by the first GitHub run, where a 0.00005
  excess on noise read as a public mismatch.
- `run_matrix.py` writes one report in the deeprelay benchmark schema
  (schema_version 1) and exits 0 / 1 / 2 (judged, nothing judged,
  mismatch). `publish.py` posts it. `verify.yml` runs on a schedule and on
  dispatch only.
