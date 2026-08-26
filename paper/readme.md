# paper/

The TMLR paper draft and its supporting math derivations.

- **`latex-tmlr/`** — the paper's LaTeX source (`main.tex`, `main-accepted.pdf`, TMLR style files
  `tmlr.sty`/`tmlr.bst`, `fancyhdr.sty`, bibliography `main.bib`). Title:
  *"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds."* Build with
  `latexmk -pdf main.tex` from inside this directory.
- **`paper-skeleton.md`** — the paper's section outline, mapped directly onto the `src/` module
  structure (§3 simulation, §5 manifold, §6 HMM, §7 drift, §8 RL).
- **`math/`** — scratch/misc math notes.
- **`math-supplementry/`** — SageMath derivations for the paper's math appendix
  (`sagemathappendix.ipynb`, `math-appendix/`), run manually via SageMath, not wired into the
  Python pipeline.

The synthetic reinforcement-learning agent (`AdaptiveAgent`) used throughout `notebooks/01`–`07`
is the paper's controlled Phase 1 instantiation of the broader hypothesis — that latent behavioral
structure and its temporal evolution in adaptive, autonomous, sequential systems occupies a
structured low-dimensional manifold — not the definition of that hypothesis's scope.
