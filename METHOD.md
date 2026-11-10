# Method

Harness version `2026.11.10`. This document is the full method, the
calibration numbers behind the thresholds, the changes made on each
version and why, and an honest account of what the test can and cannot
exclude.

## The question

Is the deployment behind `deeprelay/<model>` serving the published weights,
at comparable precision, and not a cheaper, quantized or substituted build?

## The judge since 2026.11.10: Model Equality Testing

The verdict is given by a published, peer-reviewed test, Model Equality
Testing (Irena Gao, Percy Liang, Carlos Guestrin, "Model Equality Testing:
Which Model Is This API Serving?", ICLR 2025,
[arXiv:2410.20247](https://arxiv.org/abs/2410.20247)), run through the
authors' package [`model-equality-testing`](https://github.com/i-gao/model-equality-testing)
(`pip install model-equality-testing`, version 0.0.2) exactly as
published. `model_equality.py` collects samples and applies a decision
rule; every statistic and p-value in the report comes out of the package.

**What is measured.** Every lane answers the same sixteen prompts
(`SAMPLE_PROMPTS`) at `temperature 1`, `top_p 1`, fifteen times each. The
package's `CompletionSample` holds a lane's completions as unicode
codepoints, truncated or padded to L = 200 characters (its README
setting). For two lanes the statistic is `mmd_hamming`: the maximum mean
discrepancy with the Hamming kernel (the fraction of positions at which
two completions agree, restricted to completions of the same prompt).
The p-value is the package's `permutation_pvalue` with b = 100
permutations (resolution 0.01). The paper's rejection level is alpha =
0.05: p < 0.05 means the two lanes are not sampling the same
distribution.

**Why samples at temperature 1.** The test compares whole output
distributions, so it needs none of the things the logprob fingerprint
needs: no `logprobs`, no greedy determinism, no agreement on
`top_logprobs` width. A vendor whose API offers no log-probabilities
(Z.ai, Moonshot) is a full reference here, and a thinking-only model is
compared on the visible answer it samples after thinking.

**Why the band, not the bare p-value.** Two deployments of the same
weights on different serving stacks do not produce identical
distributions, and this test is sensitive enough to say so. Measured on
2026-11-10, GLM 5.3 on Together, Fireworks and Z.ai (240 samples each)
rejected one another pairwise at p < 0.05 with MMD 0.018 to 0.022, while
every GLM 5.3 lane sat 0.066 to 0.10 from every GLM 5.3 Flash lane, p =
0.000 on all nine pairs. The paper reports the same thing for Llama 3
endpoints (11 of 31 differed from the reference weights). A bare "p >=
0.05 against a reference" would therefore fail the relay on days it
reproduces one host's distribution to within that host's distance from
the vendor's own API. So, exactly as the logprob fingerprint does, the
distances between the reference deployments define the **band** of "same
weights, different host", and the controls prove the gap to a different
model is wide.

**Verdict rule** (`model_equality.judge`), for a subject lane with at
least two usable reference deployments:

```
band        = max MMD over every pair of reference deployments
subject matches reference r  iff  p(subject, r) >= 0.05  or  MMD(subject, r) <= band
control c separates          iff  MMD(c, r) > band for every reference r
                                  and MMD(subject, c) > band and p(subject, c) < 0.05
```

- **same_model**: the subject matches its *nearest* reference (smallest
  MMD) and MMD(subject, r) < MMD(subject, c) for every reference r and
  every gating control c (closer to every reference than to any
  different model).
- **mismatch**: the subject matches no reference and MMD(subject,
  nearest reference) >= MMD(subject, nearest gating control): as far
  from the weights as a different model is.
- **inconclusive**: anything between; and the whole run is inconclusive
  when no gating control separates, because the test has not been shown
  to discriminate on that day's samples. An `advisory` control is
  reported and noted, never gates.
- **insufficient**: no subject lane with enough samples, or fewer than two
  reference deployments. A lane needs 80% of its planned samples and at
  least five on every prompt, or it is dropped and listed.

Matching the nearest reference, not every reference, is deliberate: the
band is the maximum over three pairs, so a subject that reproduces one
host's distribution can sit a little outside the band against another
host (GLM 5.3 on 2026-11-10: 0.010 and p = 0.21 to Together, 0.014 and p
= 0.11 to Z.ai, 0.029 and p = 0.00 to Fireworks, band 0.022) while being
three times closer to every reference than the nearest different model
(0.069). The control guard is what keeps that lenient: a subject can
never read `same_model` while sitting as far from a reference as a
control does.

**Prompt design.** The sixteen prompts are nine continuations of short
passages written for this harness (English, Spanish, French, German,
Russian, Chinese), three open writing tasks, two code tasks and two short
explanations. Two rules came out of the first measurement: never use a
passage the model has memorised, and tell the model not to repeat the
passage. A famous article opening (the first version used the Wikipedia
opening on photosynthesis) made every host recite nearly the same text,
and a one-word difference in which memorised phrasing a host's sampler
favoured became an MMD of 1.3 between two deployments of the same
weights, because the Hamming kernel is position-sensitive. Whether a
host's chat template makes the model restate the prompt before continuing
is likewise a template difference, not a weights difference; the
instruction removes it. With these rules the reference band for the GLM
models fell from 0.059 to 0.022 (GLM 5.3) and 0.015 (GLM 5.3 Flash) while
the sibling gap stayed at 0.064 to 0.10.

**What it cannot exclude.** The test compares distributions over the
first 200 characters of the visible answer. A substitution that leaves
that distribution unchanged (an identical model behind a different
label, a quantisation whose sampling distribution matches to within the
hosts' own spread) is not excluded; neither is anything that only shows
in long outputs. The Berkeley audit of this problem ("Are You Getting
What You Pay For?", [arXiv:2504.04715](https://arxiv.org/abs/2504.04715))
finds text-only tests weaker than logprob analysis against subtle
substitution, which is why the logprob fingerprint below still runs and
can veto, and finds that only attested execution can prove serving
integrity, which no upstream in this matrix offers.

**The logprob fingerprint** that follows is the method this harness
started with. Since 2026.11.10 it corroborates: it runs on every host that
returns log-probabilities, its full report and its own verdict
(`logprobs_verdict`) are published, a logprob `mismatch` vetoes the
sampling test's pass, and a logprob `inconclusive` or `insufficient`
changes nothing.

## The idea (the logprob fingerprint)

At temperature 0 the vector of top-k log-probabilities at each output
position is a tight function of the weights plus the serving precision. Two
deployments of the same weights stay close at every position; a different
checkpoint, a quantized build or a different model drifts, even where the
top token agrees. So the test is two-sample: the deeprelay lane is judged
against the distance between two *independently operated* deployments of
the same weights, not against a hand-set "close enough" number.

## Lanes

Each model in `matrix.json` has:

| role | what it is | judged |
|---|---|---|
| `subject` | the deeprelay model id through `https://api.deeprelay.ai/v1`, called exactly as a customer would, asking for `top_logprobs: 5`; a 400 rejecting `top_logprobs` steps the probe down to 1, then to no logprobs, and the width served is reported | yes |
| `reference` (two) | an independently operated deployment of the same weights with `top_logprobs: 5` | no: their mutual distance is the same-model band |
| `control` (one or two) | a deliberately different model: a sibling checkpoint where one is served, or a different family | no: it must separate from the references, or the metric has not been shown to discriminate |

Every lane of a model is sent the same body: the same prompts, `temperature
0`, `top_p 1`, `seed 42`, `logprobs: true`, the model's `max_tokens`, and the
model's `request_extra` (for example `{"reasoning_effort": "none"}`) so that
every lane conditions on the same thing. A lane may add its own
`request_extra` on top for a provider-specific switch (Alibaba's
`enable_thinking: false`); a `null` value removes a key for that lane. A
lane marked `stream: true` is sent `stream: true` with
`stream_options.include_usage` and its deltas are folded back into the
non-stream response shape, so a provider that serves a model stream-only can
still be a reference.

Models the relay cannot serve at temperature 0 are not listed: the method
compares greedy paths, and a lane whose upstream is pinned to sampling
cannot be compared to a greedy reference. Kimi K3 is the current example
(the relay pins its temperature to 1.0 for that upstream).

## Prompts

Twelve fixed prompts, chosen for short deterministic answers (cost) plus a
few that stress the tokenizer and the chat template: a one-word fact,
arithmetic, a string with NBSP, an emoji, a glitch token and a zero-width
space, a Chinese sentence, a code one-liner, a memorized quotation, a JSON
echo, a syllogism, a reversed rare word, a date fact, a Cyrillic/Japanese/
Turkish/Czech echo, and a self-identification question. Two replicates per
prompt on subject and reference lanes (to measure each lane's own
run-to-run noise), one on controls.

## Measurements

### 1. Content-token alignment (new 2026-11-03)

Reasoning models complicate the logprob stream. For the same request, one
provider returns logprobs over the content tokens only while another
returns them over reasoning + content, and most append their stop token(s)
after the content. Comparing the raw streams aligns a reasoning token
against a content token and fails at the first position.

So before anything is measured, each response's logprob stream is sliced to
the window of tokens whose concatenation equals `message.content`: the last
such window (content follows reasoning), with whatever precedes it (the
reasoning, a closing think tag, blank lines) and whatever follows it (stop
tokens, an empty EOS token) dropped. A token that straddles the window edge
is dropped as a partial boundary token. When nothing spells the content
verbatim (whitespace trimmed, unicode re-encoded) the content's length is
matched from the end of the stream minus trailing special tokens. A
response whose content is empty (reasoning exhausted `max_tokens`) yields
no steps and counts as *not aligned*, never as a distance of 1.0. The
report records per lane how many prompts had reasoning tokens in the
stream and how many had empty content.

Reasoning tokens are never compared. Where a model can have thinking turned
off (`reasoning_effort: none`) the matrix turns it off on every lane, so
`prompt_tokens` and the stream are content-only everywhere. Where it cannot
(GLM 5.3 and GLM 5.3 Flash reject it), thinking stays on everywhere with
`max_tokens` 2000 and only the content segment is compared.

### 2. The tokenizer/template signature, as a relative shape (changed 2026-11-03)

`prompt_tokens` reported for each of the twelve prompts fingerprints the
tokenizer plus the chat template. Two deployments of the same model agree.
A quantized build of the same model *also* agrees, so the signature can
prove a mismatch but never a match on its own.

It is now compared as a **relative shape**: each lane's count minus that
lane's own count on the anchor prompt (`p01`; the first prompt the lane
returned if `p01` is missing). A provider that prepends a fixed system
prompt (Fireworks adds a constant +3 to every Kimi K3 request) still agrees
on the shape, while a different tokenizer still shows up in the per-prompt
deltas on the CJK, Cyrillic and tokenizer-stress prompts. The absolute
counts and each lane's `signature_offset` are kept in the report.

A subject whose shape differs on **any** prompt is a `mismatch` outright. A
control separates by signature when it differs on at least **2 of 12**
prompts (`control_signature_separation` 1/6): the same rule that fails a
subject, with one prompt of slack against a flaky count. The old threshold
of 50% was calibrated for absolute counts, where a different tokenizer
differed on every prompt. Relative shapes absorb the constant, and two
English-efficient BPE tokenizers agree on most English prompts: measured on
2026-11-03, gpt-oss-120b's shape differs from DeepSeek V4.1 Flash on 4/12
prompts and from GLM 5.2 on 1/12.

### 3. Distribution distance, width-aware (changed 2026-11-03)

For each prompt and each pair of lanes, over the common greedy prefix of the
two content segments (positions after the first differing emitted token
condition on different text and are dropped), the Jensen-Shannon distance
between the renormalized top-k distributions, averaged over positions and
then over prompts. A token absent from one side is floored at that side's
smallest observed logprob, which understates disagreement rather than
inventing it. Prompts with no common prefix carry no distance: they are
counted as unaligned and excluded, never scored as 1.0 (a 1.0 in the band
or the noise floor would widen the tolerance).

Widths differ across lanes (the relay caps `top_logprobs` at 1 or 5 per
model and sometimes returns 2 when asked for 1), so at every position both
sides are truncated to the common k (each side's k most probable tokens, at
most 5) before the distance. A lane is judged at its own width `k_l =
min(width, 5)` and the reference band for that lane is computed from the
reference pairs truncated to the same `k_l`. At k = 1 the JS distance is
degenerate, so the chosen token's `|delta logprob|` (`lp_delta`, evidence
`top1_logprob`) is used instead with the same rules. The report's `band`
is the metric-to-float map at the subject's k; `bands_by_k` carries every
width from 1 to the references' width.

### 4. Noise floor

Within each lane, replicate 0 vs replicate 1 of the same prompt, over
aligned prompts, at the lane's width. A lane's distance to the references
is only meaningful against this floor.

### 5. Text lanes (new 2026.11.04.1)

A deployment whose API returns no log-probabilities cannot define or be
judged against a band. Z.ai's own API is the case today: it serves GLM 5.2,
5.3 and 5.3 Flash at temperature 0 (and turns thinking off for 5.2 with
`reasoning_effort: none`) but ignores `logprobs`. Such a lane is declared
with `"evidence": "text"` in the matrix and probed exactly like the others.
Its greedy content and relative prompt_tokens signature are compared with
every lane, and the agreement counts are reported (`text_references` in the
JSON, section 4b of the human report, one note per subject lane). Nothing
in the verdict reads them: a text lane never counts toward the two logprob
references a verdict needs, never widens or narrows a band, and never turns
an `insufficient` into anything else. It exists so a reader with the
vendor's key can see the relay reproduce the vendor's own answers, which is
weaker evidence than a distribution match (short answers agree across many
models) and is labelled as such. Measured 2026-11-04: Z.ai at temperature 0
reproduces its own answers on only 9/12 prompts for GLM 5.2 and 6/12 for
the thinking-only GLM 5.3 and 5.3 Flash, so its content agreement with the
relay (9/12, 6/12, 7/12) sits at the level of the controls (7/12). The
relative prompt_tokens shape is the part that carries information: 12/12
with the relay and the logprob references for every GLM model, 9/12 for
the MiniMax control.

Moonshot's API for Kimi K3 refuses `logprobs` and accepts only temperature
1, and the relay accepts only temperature 1 for that model too, so Kimi K3
has no greedy path on either side and is not in the matrix.

## Verdict rules

For each subject lane, with `band` the largest reference-pair distance on
the lane's metric at its width and `floor` the largest replicate noise
among the references and the lane:

```
limit = 1.5 x max(band, min(floor, 2.0 x band))
```

Noise may raise the limit by at most 2x the band: a noisy substitute cannot
buy itself a wide limit with its own jitter. A lane whose noise exceeds
that cap and lands outside the capped limit is `inconclusive` (noise and
substitution cannot be told apart), never a pass. The limit sits under an
absolute ceiling (0.20), and the band under 0.20 / 1.5 = 0.1333; a band
over its ceiling means the references disagree too much to define a
same-model band and the run is `inconclusive`.

- signature shape differs on any prompt: **mismatch**
- fewer than 10 prompts share a reference signature, no logprobs, fewer
  than 6 prompts aligned with a reference, band or limit over ceiling:
  **inconclusive**
- distance <= limit: **same_model**
- distance > limit at width >= 2 (JS distance): **mismatch**
- distance > limit at width 1 (`lp_delta`, evidence `top1_logprob`):
  **inconclusive**, never mismatch. Width 1 sees only the chosen token's
  logprob, and two deployments of the same weights differ there by about
  as much as one deployment differs from itself run to run (measured on
  GitHub's runners 2026-11-03: subject 0.0184 and 0.0161 to the
  references, band 0.0062, self-noise 0.0115, limit 0.0173). An excess
  over the limit at that width cannot be told from noise, so top-1
  evidence may confirm a match but may never fail a lane on its own. A
  `mismatch` needs decisive evidence: the signature, or a distribution
  distance at width >= 2 over the limit.
- distance > limit while thinking was on (the subject or a reference
  thought before answering on at least half the prompts): **inconclusive**,
  never mismatch. With thinking on, each lane's content tokens condition
  on that lane's *own* reasoning text, so the distance mixes reasoning
  divergence with weights and cannot be read as a weights mismatch. The
  signature still fails such a lane. `thinking_prompts` is reported per
  lane (reasoning tokens in the stream, or a `reasoning_content` beside
  content-only logprobs).

Negative controls are judged against the **largest limit any subject lane
received** on each metric, at the width that limit was computed at, so a
subject can never pass while sitting as far from the references as a
control. A control separates by the signature (2 of 12), or by every metric
a verdict relied on (distance above the limit, or fewer than 6 aligned
prompts, since a lane looking like that would itself be inconclusive).
Every gating control must separate, and each metric a pass relied on must
be separated by at least one control; otherwise the run is
**inconclusive**, never a pass.

A control marked `advisory` in the matrix measures the test's *resolution*
(a sibling checkpoint at the edge of what the metric resolves). It is
probed and reported every run, with a `note` when it does not separate,
but it does not gate the verdict; at least one gating control is still
required.

Coverage guards: a lane whose key is unset is skipped loudly; no subject
lane, fewer than two references with top-k logprobs on 10 of 12 prompts, or
a subject returning fewer than 10 of 12 prompts is **insufficient**. A
429, 5xx or transport error is retried four times with growing backoff;
a probe that still failed is sent once more after the whole sweep, minutes
later, which is what an overloaded host needs (`reprobed` on the record,
`source.reprobed` on the report). A prompt counts toward coverage if any
replicate came back.

### Reference independence (new 2026-11-03)

A reference the subject reproduces byte-for-byte (identical content on
every compared prompt, every prompt aligned, identical signature shape) at
a distance within that reference's own replicate noise (1.5x its noise, at
least 1e-4, and never more than 0.004) is the subject's own upstream, not an
independent deployment. The band it would define is "relay == upstream",
which any build of any weights satisfies. Such a reference is reported
under `dependent_references`, excluded, and the model is `insufficient` if
fewer than two independent references remain. Measured on 2026-11-03
with one candidate GLM 5.2 reference that turned out to be the relay's own
upstream: 12/12 identical at `|dlp|` 4e-8, 7e-7 and 1.6e-3 across runs
(that deployment's own jitter 0 to 2.3e-3). Independent same-weights pairs
never measured under 0.006. The candidate was dropped from the matrix. The 0.004 ceiling exists because a noisy reference
(Fireworks measured a 0.017 jitter in one run) must not widen "the same
deployment" until an independent subject at 0.023 qualifies.

## Calibration

Same-model bands at k = 5 measured across 2026-10-12, 2026-10-14 and
2026-11-03: JS 0.019 to 0.057; chosen-token `|dlp|` at k = 1: 0.011 to
0.055. Replicate noise floors 0.002 to 0.08 (thinking-on lanes are the
noisy ones). Different-family controls that fail to align at the first
content token on most prompts score JS 1.0 or carry no distance; the ones
that do align score JS 0.07 to 0.15 and `|dlp|` 0.07 to 0.13.

Thresholds (`thresholds` in every report): `js_tolerance` 1.5,
`noise_cap_x` 2.0, `js_limit_ceiling` and `lp_limit_ceiling` 0.20, band
ceilings 0.1333, `min_compared` 10, `min_aligned` 6,
`control_signature_separation` 0.1667, `topk_cap` 5, `dependent_distance`
1e-4.

## What the test cannot exclude (measured 2026-11-03)

Honesty about the limits matters more than the pass count.

- **At k = 1 the chosen-token metric cannot be reliably shown to
  discriminate.** A different family that is equally certain of the short
  answers lands inside the limit: gpt-oss-120b answers "Paris", "391",
  "29", "Yes" with the same near-zero logprob as DeepSeek V4.1 Flash or
  GLM 5.2 (chosen-token distance 0.004 to 0.007 against limits of 0.02 to
  0.05; relative signature differing on only 1 to 4 prompts), so it is not
  used as a control for the k = 1 models. MiniMax M3 separated from the
  DeepSeek references at 0.069 and 0.070 in two runs and then at 0.012 in
  a third (it happened to agree on the certain tokens of the 8 prompts
  where it aligned), so even the control that usually works does not
  always. The pre-1017 sibling checkpoint sits at the edge: 0.025 to
  0.033 against limits of 0.022 to 0.029 across four runs, which is why it
  is advisory. Consequence: at the relay's current `top_logprobs: 1` cap,
  DeepSeek V4.1 Flash reads `pass` on some runs and `inconclusive` ("no
  control separates on lp_delta") on others while its lane measures
  `same_model` with signature 12/12 every time. The fix is on the
  deeprelay side: raise the relay's `top_logprobs` cap from 1 to 5 for
  DeepSeek V4.1 Flash, GLM 5.2 and GLM 5.3 Flash. At k = 5 MiniMax measured
  JS 0.043 to 0.087 against DeepSeek limits of 0.025 to 0.041.
- **Thinking-only models are only weakly testable.** GLM 5.3 and GLM 5.3
  Flash cannot be turned off, and with thinking on each lane's content
  conditions on its own reasoning, so cross-lane distances carry reasoning
  divergence: the deeprelay GLM 5.3 lane measured JS 0.050 and 0.069 in
  two runs against limits of 0.076 and 0.066, flipping across the line,
  while the GLM 5.3 Flash sibling control sat at 0.030 and 0.036, inside
  the limit both times (and `lp_delta` 0.016 to 0.023 at k = 1 for the
  reverse pairing). So neither model can be failed on distance (the
  harness downgrades such a result to `inconclusive`), neither can pass
  while its sibling control is not separated, and both read
  `inconclusive`. A pass obtained by dropping the sibling control would
  be an overclaim.
- **Qwen 3.8 Flash's two references disagree by JS 0.056** (Alibaba vs
  Together), close to what a different family scores (0.08 to 0.09), so the
  controls sit on the limit: gpt-oss separates at 0.093 vs 0.088, MiniMax
  does not at 0.083. Runs will read `inconclusive` whenever either control
  lands inside.
- **The signature is weaker than it was.** Relative shapes were the price
  of accepting a reference that prepends a system prompt. A subject with a
  different tokenizer still mismatches (one differing prompt is enough),
  but controls need the distribution evidence more often than before.
- A lane that does not honour `temperature: 0` (an upstream that forces
  sampling) cannot be tested by this method at all, whatever its weights:
  its path is sampled and its reported top-k is not the greedy one. Such
  models are left out of the matrix (Kimi K3 today).

## Verdicts on 2026-11-03

Four full matrix runs were made while calibrating (about 2,300 requests).
The table is the third run, re-judged under the final rules; where a later
run differed, it says so.

| model | verdict | subject | reasoning |
|---|---|---|---|
| DeepSeek V4.1 Flash | **pass** | same_model | signature 12/12; `lp_delta` 0.0154 vs limit 0.0293 (band 0.0195); MiniMax separates at 0.0697; the advisory pre-1017 sibling at 0.0250 does not. Subject `same_model` in 4/4 runs (0.0066 to 0.018); run 4 read `inconclusive` because MiniMax's `lp_delta` fell to 0.012 (see the limits section) |
| GLM 5.2 (out of the matrix since 2026.11.04) | **insufficient** | insufficient | only one independent reference with logprobs serves GLM 5.2 (Together); Fireworks does not carry it and the vendor's own API does not return logprobs; a candidate second reference was found to be the relay's own upstream (12/12 identical, distance 4e-8 to 1.6e-3) and dropped |
| GLM 5.3 | **inconclusive** | same_model | JS 0.020 vs limit 0.070 this run (0.050 and 0.069 in earlier runs, once over the limit, downgraded to inconclusive by the thinking-on rule); the GLM 5.3 Flash sibling control sits at 0.030 to 0.039 and never separates; the deeprelay lane self-identified as "YuanYuanLLM" on the self-ID prompt in one run |
| GLM 5.3 Flash | **inconclusive** | same_model | `lp_delta` 0.018 vs limit 0.088 (band 0.059); the GLM 5.3 sibling control sits at 0.016 to 0.023 and never separates |
| Qwen 3.8 Flash | **pass** | same_model | JS 0.057 vs limit 0.080 (band 0.042); gpt-oss separates at 0.111, MiniMax at 0.100. Inconclusive in run 1 (MiniMax 0.083 vs 0.088), pass in runs 2 to 4: the references disagree by JS 0.04 to 0.06, so the controls sit near the limit |

Kimi K3 was in the calibration matrix and measured outside the band in
4/4 runs; the cause is that the relay pins its temperature to 1.0 for that
upstream, so a temperature-0 comparison is not possible through the relay
and the model is not listed. That is a limit of the method, not a finding
about the weights.

Model-level reasoning is in each run's report; the table above is the
2026-11-03 calibration and will go stale.

## Changes on 2026-11-10 and why

The sampling test (Model Equality Testing, above) becomes the judge and
the logprob fingerprint the corroborator. Why: with the logprob
fingerprint alone, GLM 5.3 and GLM 5.3 Flash read `inconclusive` on every
daily run (their subject lanes `same_model`, but the sibling control could
not be told apart on the content-token logprobs: JS 0.030 vs limit 0.046
and 0.055 vs 0.119), Z.ai could only be a text lane, Kimi K3 could not be
listed at all, and the method was deeprelay's own. The published test
needs samples rather than logprobs, which brings every vendor API into
the band, and it separates the siblings decisively.

Measured before shipping, 2026-11-10, 240 samples per lane (sixteen
prompts, fifteen each), b = 100, L = 200:

| pair | GLM 5.3 | GLM 5.3 Flash |
|---|---|---|
| relay vs Together (MMD, p) | 0.010, 0.21 | 0.0045, 0.29 |
| relay vs Fireworks | 0.029, 0.00 | 0.0031, 0.31 |
| relay vs Z.ai | 0.014, 0.11 | 0.0035, 0.27 |
| band (max reference pair) | 0.022 | 0.015 |
| relay vs sibling on Fireworks | 0.088, 0.00 | 0.064, 0.00 |
| relay vs sibling on Together | 0.069, 0.00 | 0.070, 0.00 |
| relay vs sibling on Z.ai | 0.070, 0.00 | 0.089, 0.00 |
| sibling vs nearest reference | 0.066 | 0.066 |

Both read `same_model` with all three sibling controls separating. The
first version of the prompt set, with a memorised Wikipedia opening and
no instruction against restating the passage, gave bands of 0.059 and
0.059 and sibling gaps of 0.091 and 0.094 (see "Prompt design"); the
shipped prompts are the second version.

Also in this version: `requirements.txt` and the package install in both
workflows (CI on Python 3.11 to 3.13, since torch 2.15 needs 3.11); the
matrix gains `defaults.samples` and a per-model `samples` block (GLM 5.3
and GLM 5.3 Flash sample at `reasoning_effort: low`, since their thinking
cannot be disabled and would otherwise exhaust the token budget); a lane
marked `stream: true` is sampled over SSE too; `run_matrix.py --no-samples`
runs the logprob fingerprint alone; a text lane the sampling test judged
is a wire `reference`.

## Changes on 2026-11-06 and why

- Retries and a re-probe pass. The scheduled run of 2026-11-06 read
  DeepSeek V4.1 Flash `inconclusive` with its subject lane at `same_model`
  (`lp_delta` 0.025 vs limit 0.047): Fireworks answered "service
  overloaded" on 3 of 12 prompts of MiniMax M3, the model's only gating
  control, two retries over 6 s did not outlast the burst, the control was
  dropped at 9/12 and no gating control was left to demonstrate
  separation. A probe is now retried four times with growing backoff
  (2, 4, 6, 8 s), and every probe that still failed on a retryable error
  is sent once more after the whole sweep, minutes later. The first local
  run with this code re-probed one MiniMax probe on GLM 5.3 Flash that had
  failed all four retries and recovered it. A re-probed record is marked
  (`reprobed`, `first_error`) and counted in `source.reprobed`, so a
  report shows how much of its coverage the second pass bought.
- A second gating negative control on a different host for every model
  that had one. A single gating control on a single host is a single point
  of failure for the whole verdict. Measured 2026-11-06 (local run, all
  references and controls at 12/12): Qwen 3.8 Flash on Together separates
  from the DeepSeek V4.1 Flash references by signature on 7/12 prompts
  (`lp_delta` 0.051 vs limit 0.060 at k = 1 does not separate on its own);
  DeepSeek V4.1 Flash 1017 on Together separates from the GLM 5.2, GLM 5.3
  and GLM 5.3 Flash references by signature on 3/12 and by JS distance
  (0.060 vs limit 0.098 and 0.045 vs 0.066 at k = 5 for the two GLM 5.3
  models). Qwen 3.8 Flash gets the same model as a third control: its two
  controls sit on the limit (gpt-oss 0.086 and MiniMax 0.077 vs 0.082 and
  0.095 across the day's two local runs), and a run on which neither
  clears it reads `inconclusive` for want of a control that separates on
  the metric; DeepSeek V4.1 Flash 1017 measured JS 0.103 vs limit 0.082
  with signature differing on 7/12. The verdicts did not move: DeepSeek
  V4.1 Flash `pass`, Qwen 3.8 Flash `pass`, GLM 5.2 `insufficient`, the
  GLM 5.3 pair `inconclusive` on the sibling control as before. The
  subject lanes, references, prompts, thresholds and metrics are
  unchanged, so the version bump marks the measurement loop.

## Changes on 2026-11-04 and why

- DeepSeek's own API (`api.deepseek.com`, model `deepseek-flash` with
  `reasoning_effort: none`) returns top-5 logprobs and joins the DeepSeek
  V4.1 Flash references, so the vendor's own deployment is in the band.
- Z.ai joins as a text lane for the three GLM models (see Measurements 5).
  GLM 5.2 is back in the matrix with Together as its one logprob reference
  and reads `insufficient` until a second one exists; the text lane shows
  what the relay's answers look like next to the vendor's.
- The scheduled run is daily. The mark on deeprelay.ai reflects the latest
  daily run and is withdrawn by the API when the latest run is older than
  14 days.

## Changes on 2026-11-03 and why

The 2026-11-03 re-probe of the private harness found two defects: (1) the
DeepSeek V4.1 Flash 1017 checkpoint thinks by default, Together returns
logprobs over content only while Fireworks and the relay return them over
reasoning + content, so first-token alignment failed Together against
everything and left one reference and no band; (2) both negative controls
had been retired upstream. The fixes above (content alignment, per-model
request extras, streaming lanes, relative signature, width-aware
distances, independence check, two retries) are what this version is. The
harness version was bumped so that older reports are not read as
comparable.
