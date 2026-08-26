# outputs/reports/

Per-notebook LaTeX/PDF writeups, one directory per notebook (`nb01`–`nb07`), each with its own
`lbsm_notebookNN.tex`/`.pdf`: full methodology, numerical results, figures, evidence checklist,
findings, and significance-for-the-paper discussion, sourced directly from that notebook's cell
outputs and exported CSVs. `nb01`–`nb04` currently have only the compiled PDF (no `.tex` source
was ever committed); `nb05`–`nb07` have both.

- **`lbsm_unified_report.tex`/`.pdf`** — all seven reports merged into a single document, each
  notebook as its own numbered section, with per-notebook export/figure listings consolidated into
  one unified file-manifest appendix at the end.
- **`issues/`** — pre-registered critical-issue notices written and resolved before the notebook
  they concern was executed (currently: `lbsm_covariance_issue.pdf`,
  `LBSM-ISSUE-NB07-001`, on full-covariance HMM numerical instability, resolved in
  `src/hmm/robust_fitting.py` before NB07 ran). `lbsm_covariance_issue.pdf` is git-ignored by
  name — kept locally, not checked in.
