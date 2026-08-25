# NB08 — Real-World Generalization of LBSM

Status: planning, not started. No code, data, or notebook exists yet for this work.

## Why this notebook exists

LBSM's hypothesis is **latent behavioral structure and its temporal evolution in
adaptive, autonomous, sequential systems** — not "RL agent telemetry" specifically.
Notebooks 01–05 test that hypothesis inside a controlled synthetic construction
(`AdaptiveAgent`, 4 known hidden regimes, RL-driven adaptation) precisely because a
controlled setting gives known ground truth to validate the methodology against.

NB08 is the second, real-world test of the same hypothesis, not a bonus/future-work
add-on and not an attempt to force real telemetry into the synthetic agent's four
labels (`stable`/`exploratory`/`adaptive`/`unstable`). Its job is to test whether the
**machinery** — manifold projection, regime/structure recovery, drift and anomaly
detection, transition analysis — remains meaningful when the observation domain
changes to a system with no constructed ground truth.

```text
                 LBSM
                   |
       latent behavioral/state
          temporal structure
                   |
       +-----------+-----------+
       |                       |
Synthetic adaptive        Real sequential
agent telemetry              systems
       |                       |
       |                 +-----+-----+
       |                 |           |
   known regimes       MPF         REMS
       |
       v
controlled validation
```

Target real datasets: **MSL-REMS** (Curiosity rover Rover Environmental Monitoring
Station) and **MPF rover engineering telemetry** (Sojourner microrover's
engineering sensors — communication/health/error counters, motor and wheel
fault counts, battery/CPU/motor temperatures; confirmed from the PDS label as
`DATA_SET_ID = "MPFR-M-RVRENG-2/3-EDR/RDR-V1.0"`, *not* the ASI/MET
atmospheric instrument as originally assumed here — see `datasets/README.md`),
both public NASA PDS products. This MVP scopes **MSL-REMS only** — see
"Deliberately excluded" below for why MPF and the non-drift branches are deferred.

## What stays invariant vs. what changes

Moving to real data does not remove these five methodological constraints — they
become part of the experimental design rather than reasons to exclude the datasets:

1. **No ground-truth regime labels.** The NB03 Hungarian-alignment accuracy metric
   cannot be computed on REMS. Any regime/structure claims need unsupervised
   selection (e.g. BIC/AIC over `n_components`) validated against documented mission
   events as a weak external reference, not a hard accuracy number.
2. **Domain-specific feature extraction.** `latency/entropy/reward/action_freq` from
   `src/simulation/behavior_profiles.py` doesn't transfer. REMS needs its own
   physical feature representation (pressure, temperature, wind, etc.).
3. **Stationarity / seasonality.** REMS has strong diurnal and Ls (seasonal) cycles
   that must be separated from genuine drift before the healthy-envelope branch
   means anything — the synthetic generator has no analogue to this problem.
4. **HMM/Gaussian covariance assumptions.** Real physical channels are
   correlated (e.g. pressure/temperature co-vary with weather); the synthetic
   `covariance_type="diag"` assumption should be treated as untested here, not
   inherited by default.
5. **Population vs. single sequence.** The statistical unit changes from
   20 independent agents x T timesteps to one long mission sequence. Claims about
   between-agent variance (used throughout NB01-05) don't transfer as-is.

## What already reuses cleanly

Checked against the actual code in this session:

- `src/drift/ewma.py` (`fit_ewma`, `ewma_scores`) and `src/drift/kl_divergence.py`
  (`fit_reference`, `kl_drift_scores`) operate on plain `(T, d)` / `(N, d)` arrays
  with no dependency on ground-truth regime columns — usable on REMS residuals
  unmodified.
- `src/drift/drift_detection.py::fit_healthy_envelope()` is the one exception: it
  selects "healthy" rows via `regime_col="hidden_state"` filtered against
  `healthy_regimes=("stable","exploratory","adaptive")`, which doesn't exist for
  real data. This needs a small additive change (see Phase 4).
- `configs/drift.yaml` does not currently exist at all (the drift module is
  presently unconfigured / parameters live in notebook cells) — a new REMS config
  isn't overriding anything, it's the first config file for this module.

## Task list

### Phase 0 — De-risk the core assumption (do first, before any pipeline code)

- [x] Confirm MSL-REMS data is retrievable in usable form from the NASA PDS
      Geosciences/Atmospheres node. **Resolved**: dataset is
      `MSL-M-REMS-3-TELRDR-V1.0` (REMS RDR SIS, PDS Atmospheres catalog). The
      processing chain is EDR (raw counts) -> **TELRDR** (electrical/thermal
      units, our target) -> ENVRDR (environmental/physical units) -> MODRDR
      (corrected/modelled physical units). Filenames encode the product as a
      3-character code: `RTL`=TELRDR, `RNV`=ENVRDR, `RMD`=MODRDR,
      `ADR`=ancillary data — confirmed by cross-checking every file currently
      in `datasets/rems_sol01_89/` (see `datasets/README.md`). Only
      `RME_*RTL*_PA.TAB` is needed for NB08; ADR/RMD/RNV are separate products
      NB08 doesn't use.
- [ ] Confirm dated external reference events exist at sol/timestamp resolution
      (documented instrument-fault windows, the 2018 global dust storm period,
      calibration breaks). Without dated events there is no F1/FPR to compute —
      decide now whether an unsupervised-only fallback (structure description, no
      quantitative detection metrics) is acceptable if events are too sparse.
- [ ] **Stop/go checkpoint**: if data access or event documentation is too thin,
      park this notebook and say so explicitly rather than building the rest of
      the pipeline on an unverified foundation.

### Phase 1 — Data acquisition

- [ ] Download `RME_*RTL*_PA.TAB` (TELRDR only — see Phase 0) for a sol range
      spanning at least one documented anomalous period plus adjacent nominal
      periods. Per the REMS RDR SIS, each TELRDR row is one acquisition
      session with columns ordered: time references, Wind Sensor, Ground
      Temperature Sensor, Air Temperature Sensor, UV Sensor, Humidity Sensor,
      Pressure Sensor — confirm exact column offsets/units against the SIS
      (or a `.LBL`/`.xml` label, if one ships alongside the `.TAB` files)
      before parsing; none has been seen on disk yet as of this note.
- [ ] Write a one-off loader script (PDS label/data file pairs, not CSV — do not
      try to force this through the existing CSV-based `TelemetryGenerator` I/O
      path) that parses into a flat `(timestamp, channel_1, ..., channel_k)` table.
- [ ] Save raw parsed output to `data/raw/nb08/`, following the existing
      per-notebook data convention.

### Phase 2 — Domain feature representation

- [ ] Decide the feature set (raw channel values + short-window rolling
      statistics is the likely starting point — REMS has no natural analogue to
      latency/entropy/reward).
- [ ] Write new REMS-specific extraction logic (new module/function — do not
      modify `src/telemetry/feature_extraction.py`, which stays specific to the
      synthetic agent's feature set).
- [ ] Handle missing data / sensor dropouts explicitly (interpolate vs. drop vs.
      mask) and document the choice — the synthetic pipeline never had to make
      this call and has no precedent for it.

### Phase 3 — Deseasonalization

- [ ] Characterize the diurnal + seasonal (Ls-based) cycle in each channel.
- [ ] Remove it (fit and subtract a smooth seasonal/diurnal baseline, or work in
      residual-from-expected-value space) so the drift branch isn't just
      detecting "it's night" or "it's winter."
- [ ] Sanity-check by plotting residuals over a known-nominal stretch and
      confirming they look stationary before trusting anything downstream.

### Phase 4 — Healthy-window definition & envelope calibration

- [ ] Manually select a "healthy" calibration window (documented nominal period,
      no known faults/storms) from the deseasonalized residuals.
- [ ] Generalize `fit_healthy_envelope()` to accept an explicit boolean
      healthy-mask parameter as an alternative to `regime_col`/`healthy_regimes`
      (small additive change — the existing GT-label code path used by NB04 must
      keep working unchanged).
- [ ] Fit the envelope on the REMS residuals using that mask.

### Phase 5 — Drift/anomaly scoring

- [ ] Run `ewma.py` and `kl_divergence.py` as-is on the REMS residual series.
- [ ] Run Mahalanobis scoring via the envelope adapted in Phase 4.
- [ ] Combine via `combined_anomaly_score()` — verify it doesn't assume any GT
      columns before treating it as a drop-in.

### Phase 6 — Validation against real events

- [ ] Build the event-label vector: binary "in documented anomalous window" per
      timestamp, from the events identified in Phase 0.
- [ ] Run `threshold_sweep()` against this label vector; report best-F1 operating
      point, FPR at that point, and detection latency relative to each event's
      documented onset — the same three numbers NB04 reported, for direct
      comparison.
- [ ] Manually inspect false positives/negatives to check whether they cluster
      around genuinely ambiguous periods (e.g. minor local dust lifting) rather
      than being random — this distinction matters for how the result can
      honestly be described in the paper.

### Phase 7 — Integration

- [ ] New `configs/drift_rems.yaml` holding REMS-specific window/threshold
      parameters, following the existing per-module config convention.
- [ ] Assemble as `notebooks/08_real_world_generalization.ipynb`, following the
      existing evidence-checklist + summary-table format used by NB01-05.
- [ ] Write the corresponding paper section, explicit about scope: unsupervised
      structure description + binary anomaly detection only. No multi-regime
      ground-truth accuracy claim (that metric does not exist for this dataset).

## Deliberately excluded from this MVP (future work, not silently dropped)

- **MPF (Mars Pathfinder rover engineering telemetry)** — thinner external event documentation, much
  shorter single-lander mission. Do REMS first; revisit MPF only if REMS proves
  the approach works and event documentation for MPF turns out to be adequate.
- **Multi-regime HMM discovery on REMS** — no ground truth to validate a recovered
  regime count against. Only worth doing if BIC-selected regimes visibly correlate
  with the same documented events used for Phase 6, as a secondary check.
- **Manifold projection and RL branches on real data** — this MVP is drift/anomaly
  detection only, matching the piece of the pipeline (NB04's composite scorer)
  that is most directly transferable and has the clearest external validation
  signal (dated real events) available.

## Risk ranking

Riskiest items are Phase 0 (data/event availability — could invalidate the whole
notebook before any pipeline code is written) and Phase 3 (deseasonalization —
if this doesn't work cleanly, every downstream drift number is contaminated by
diurnal/seasonal cycles rather than genuine anomalies). Both should be resolved,
even roughly, before investing time in Phases 4-7.
