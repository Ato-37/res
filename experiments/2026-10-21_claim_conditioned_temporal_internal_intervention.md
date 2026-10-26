# Claim-conditioned temporal internal intervention: cohort preparation

## Research question

Test whether a fixed internal visual-key bias, selected from the frozen
teacher-forced claim-token attention to temporal video groups, increases the
claim's dependence on those groups more than on a disjoint, equal-budget
hash-selected sham. This is an exploratory proxy. Teacher-forced claim-token
attention can be self-conditioned and the current claims/evidence windows have
not been human reviewed; this experiment cannot establish semantic support or
direct faithfulness.

## Cohort freeze and claim generation

- Dataset: local NExT-GQA `test` candidate pool `test_claim_pool_local_20261021_03`.
- The first lock/packet (`20261021_07`) was found by independent audit to
  overlap the prior gold-interval test by **3/3 exact media+QID assignments**.
  It was not scored for this intervention and is excluded from the experiment.
- A derived pool added all 10 rows from the prior gold-interval lock to the
  candidate exclusion set (20 source-pool media; 10 QA-scored media excluded;
  111 candidate QA pairs). New lock/packet (`20261021_08`) was frozen and
  generated from that exclusion. It has 3 unique rows, zero exact assignment
  overlap with gold lock06 or lock07, and all three video byte hashes match.
  The local candidate pool had been enumerated before and exposure history is
  not complete/authenticated, so this remains exploratory rather than a fresh
  independent test.
- Cohort lock SHA-256:
  `ed1de4dbd6d6a868962ce317a6dc22eb7ea1bf419e528be108c4634a8e32be32`.
- Pair filter SHA-256:
  `29428221f42f23ad9baedf9da64193270827dfa73cd73047351a32dd139c3c98`.
- Claim-generation packet SHA-256:
  `06e44cfd1b3de36b6fcf01a1298cc884d03e0858e29a0c08044db96253d649ab`.
- Generation: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights/BF16 compute,
  eager attention, 16 frames/video, prompt v1, deterministic decoding.
- Result: 3/3 structured claims; reported runtime 16.1 s after model setup;
  no faithfulness score computed. Generation packet is local mode 0600.
- Primary checked packet-to-lock media/QID assignments and the separate ordered
  pair-filter binding: 3/3 rows match, 3/3 actual video byte hashes match, 3/3
  claims parse as structured, and packet/lock/filter are mode 0600. Raw
  questions, answers, claims and media paths remain private. Independent audit
  is pending for packet08.

## Planned treatment contrast

Capture untreated attention from the exact teacher-forced claim-token query
rows at the fixed visual layers. Aggregate attention over complete temporal
groups; choose a preregistered top-k treatment set and a stable disjoint
equal-k hash-ranked complement sham. Freeze all selector, layer, bias, and
token-score settings before inspecting intervention outcomes. Compare fixed
claim-token likelihood under original and block-permuted temporal order for
baseline, treatment and sham. The primary is the treatment-minus-sham
original-versus-permuted interaction. This tests content/order-specific
likelihood sensitivity under this proxy, not whether a claim is true or whether
its selected interval is human-valid evidence.

## Full exploratory run: frozen cohort 08

Two full-run preflight attempts correctly executed zero forwards and rejected
their smoke receipts: first because the runner source hash had changed after
the smoke, then because the validator looked for the row token at the wrong
JSON nesting level. After fixing the schema check, another full attempt reached
video preparation and failed before any forwards. A real-video CPU diagnostic
isolated Qwen's `total_num_frames=15.999999999999998` on one nominal 16-frame
sample; exact comparison rejected float roundoff. The processor check now
accepts only a tight `1e-7` relative/absolute tolerance. The failure was
reproduced on that sample, and 26 focused regression tests passed before
re-running the source-bound smoke and full experiment.

- Corrected smoke receipt SHA-256:
  `1bc0ce780e7ab4af34bb633161d7fe0de4bf0064dca57843e5d7e5ae4744ffcd`.
- Full receipt SHA-256:
  `9c2cfbff2581fe17a1331bd0a5061722a22f753351bb3594037db1ecf7b08514`;
  both receipts are local mode 0600. Full run completed 18/18 forwards on 3/3
  locked rows. Lock/filter/packet hashes match the values above; source hashes
  are recorded in the receipt. Independent result audit is pending.
- Primary outcome is the row-level treatment-minus-sham temporal-order
  interaction, in nat/token: **+0.023010, +0.033725, +0.018543**; mean
  **+0.025093**, median +0.023010, range [+0.018543,+0.033725], 3/3 positive.
- The component treatment order interactions were **−0.017952, +0.037386,
  −0.008962** (mean +0.003491; 1/3 positive). Hash-sham order interactions
  were **−0.040963, +0.003661, −0.027505** (mean −0.021602; 1/3 positive).
  Baseline natural-minus-reversed likelihood differences averaged +0.006489
  (2/3 positive). Thus the positive treatment-minus-sham contrast is partly
  driven by negative sham interactions; it is not a consistent increase in
  the treatment branch's own temporal dependence.
- The learned/generated claim is teacher-forced and self-conditioned; selection
  uses its baseline attention. Reversal is an OOD temporal perturbation, the
  cohort is exploratory with incomplete authenticated exposure history, and
  human claim/evidence review was not done. The capture receipt explicitly
  marks attention/source authentication and scoring authorization false. Treat
  this as a computational proxy pilot only: **it does not establish improved
  faithfulness, semantic support, or valid evidence localization**. No
  population inference is warranted at n=3.

The result is a useful thesis seed: a matched-sham specificity contrast can be
positive even when the treatment's own order interaction is mixed. A defensible
follow-up should use a preregistered claim-support outcome tied to independently
reviewed evidence and compare the fixed explanation against the same video,
matched sham windows, and question/prompt controls. Do not tune on this cohort.

## Frozen non-overlapping follow-up: cohort 09

To test whether the cohort08 positive specificity pattern repeated, cohort09
was selected before claim generation from the same pre-existing 20-video local
pool. The freezer's accumulated exclusion set had 156 media hashes, including
all 134 locally scored/generation-exposed media it detected; cohort09's 3 media
were absent from that set and had zero media overlap with cohort08. A separate
CPU reconstruction of the frozen selection reproduced the exact three lock
assignments and cohort digest. This is a disjoint-row exploratory replication
within a previously enumerated pool, not an independent population sample;
the prior candidate inventory and incomplete authenticated exposure history
remain limitations.

- Cohort lock SHA-256:
  `6d2ff209aa137c0ec8d33a9bd9dca00980becbc9ab583c7c830c9cd71bfc3406`;
  pair filter SHA-256:
  `f0fcb1f1f1bdab374c52af79ab6de130f72ac65b607827edae47356198d0c65e`.
- Claim packet SHA-256:
  `a5a8210bae2c69c4e6875ac754ffe0b13ebb72ade4c3f8f66d43a40bc2093576`;
  3/3 structured claims. Source pair filter was limited to the exact three
  frozen assignments. The model, prompt v1, frame count, layers, bias, selector,
  sham rule, and scoring contrast were unchanged from cohort08.
- Smoke: 6/6 forwards on 1 row, receipt SHA-256
  `8b06f05bea3f62d1a7fbca4e4112b52d2edebf8138d5c67e7004dad14f8ac33e`.
- Full: 18/18 forwards on 3/3 rows, receipt SHA-256
  `0f29878b5b5434bf4e9f6e05c56516f8ce9e96657eb6322d66871299360f0184`;
  all private artifacts are mode 0600. Runner preflight and local hash/overlap
  checks passed. Independent receipt audit confirmed current source/input
  hashes, exact row order, 18/18 forwards, equal disjoint selected/sham key
  budgets (264/264, 264/264, 252/252), reversal mapping, and layer-hook counts.
  It also reproduced the current local-scanner exclusion set (134 prior scored
  media; zero overlap). The lock records its digest/count but does not bind a
  separate exposure-inventory manifest hash; historical exposure completeness
  remains unverified. The audit checked receipt and arithmetic, not an
  independent GPU/model rerun.
- Key DID: **+0.008707, −0.000724, +0.006115** nat/token; mean +0.004699,
  2/3 positive. Sham DID: +0.015587, +0.014968, +0.003060; mean +0.011205,
  3/3 positive. Key-minus-sham specificity: **−0.006880, −0.015693,
  +0.003055**; mean **−0.006506**, 1/3 positive. The first cohort's positive
  specificity did not replicate.
- Pooled descriptive summary of the six distinct rows (no inferential test):
  key DID mean +0.004095 nat/token (3/6 positive); sham DID mean −0.005199
  (4/6 positive); key-minus-sham mean +0.009294 (4/6 positive; range
  [−0.015693,+0.033725]). This small pooled mean is unstable across the two
  cohorts and cannot support an efficacy claim. Human review remains 0/6, and
  all six attention-capture receipts still mark source authentication and
  scoring authorization false.

## Protocol authorization audit

An independent source audit of the capture contract found that these `false`
flags are an explicit authorization gate, not just a reminder that the result
is uncertain. The capture utility and its contract always emit
`forward_source_authenticated=false`, `attention_capture_authenticated=false`,
and `scoring_authorized=false`; their receipts permit zero authorized scoring.
The cohort08/09 runner checked tensor geometry and hook counts but did not reject
those false flags, so it nevertheless loaded the model, performed forwards and
computed likelihood scores. This is an implementation/protocol failure. Keep
the outputs private and preserve their measured numbers as historical
**non-authorizing exploratory diagnostics only**; do not describe them as
protocol-approved experiments, faithfulness scores, or evidence for efficacy.

Do not launch further model-scoring branches that depend on these captures until
the protocol is resolved. To resume an authorized claim-level study requires
the upstream claim/evidence and prediction-link/timing reviews, independently
pinned reviewer trust root and cohort, an authenticated capture for the pinned
model/runtime, and a versioned runner/receipt gate that verifies those inputs
before model load or scoring. Existing false booleans cannot be relabeled true.
The user may waive a manual convenience review, but that alone does not create
the missing technical/authenticated evidence. Independent audit of the
cohort09 lock also reproduced the locally known 134-media scored exposure set
and zero overlap, while noting that the broader historical exposure inventory
is not fully authenticated.

## Fail-closed runner correction (2026-10-21)

The current prototype runner now rejects `smoke` and `full` before reading the
locked packets, importing CUDA/model libraries, decoding protected video, or
making any forward. The failure receipt records
`capture_protocol_authorization_not_granted`, stage
`capture_protocol_preload_authorization`, and zero attempted/completed forwards.
CPU `preflight` remains available for structural input validation. A new
versioned runner and upstream evidence are required to enable scoring; changing
the current capture's false fields is not supported. A regression test invokes
the actual GPU entry point with a mocked receipt writer and verifies the
pre-model gate and zero counts. The focused runner/capture-contract suites
passed **47/47**, compilation passed, and `git diff --check` passed. No GPU
scoring was run. The 08/09 historical receipts and thesis synthesis must retain
their prior source hashes and be labeled non-authorizing; this guard does not
retroactively alter them.

The CLI guard was also exercised with `--mode smoke`. It returned the expected
blocked status before packet loading, with 0 rows and 0/0 forwards. The owner-
only receipt is `data/private_nextgqa/claim_token_key_order_guard_test_20261021_01.json`
(mode 0600, SHA-256
`a7c270c9749ec56da1b8abcb33eb50857b3b6dda7e2c8dcf3abac60ea6dcf1d4`). This
is a guard-path test, not a model/GPU experiment or a direct faithfulness
measurement.

The best thesis result is now a **failed replication of an exploratory proxy
effect**: a 3/3 positive matched-sham contrast in cohort08 became 1/3 positive
and negative on average in the disjoint cohort09. That is useful evidence
against treating the original proxy lift as stable. The scientific conclusion
remains limited to this model, intervention, candidate pool and exploratory
likelihood endpoint; direct faithfulness is still unmeasured.

## Next steps and limits

The runner and CPU contract tests now cover target-token offsets, video group
geometry, selected/sham disjointness, key/query causality, hook scope and
cleanup, branch counts, and receipt bindings. Cohort 08 is scored and closed;
do not rerun or tune it. The next sound step is to reconcile the explicitly
non-authorizing attention-capture status with the study's evidence protocol,
then define an outcome that measures claim support rather than generic
likelihood sensitivity. The hand-review candidate artifact is ready, but no
human responses exist for this cohort.

Rights status remains unverified. The user authorized local-only operations;
the packet will not be published or pushed.
