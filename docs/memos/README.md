# Memos

These memos derive the measurement models that eigsep_cal implements. They ship with the code, so the derivation behind each equation is available to anyone working on the package.

| Memo | What it gives the code |
|---|---|
| [M003, receiver calibration](M003_receiver_calibration/memo.md) | Eqs. F, Ps, availgain and map (`api.md` § 4.1–4.3), and eqs. Q, cal, TNSTL and X (§ 4.5). Used in `forward.power`, `network.embed`, `receiver.noise_wave_map`, and the planned stage-3 solvers. |

`api.md` and the docstrings cite equations by their names in the memo (eq. Ps, eq. map, …). The Markdown shows the same names as equation tags.

## Origin

- **Generated, not hand-written.** Each `memo.md` is produced from the memo's LaTeX source in the EIGSEP Deployment 5 analysis manuscript, a private repository, by its `scripts/memo2md.py`.
- **Source commit.** The header of each file names the manuscript commit it came from.
- **Figures.** They are PNG renders of the memo's figures.
- **References into the private workspace.** The memos were written for the D5 analysis, so some references point to material that is not in this repository:
  - notebooks such as `003_receiver_cal_formalism`;
  - `products/` paths and logbook entries;
  - question IDs (`Q-CHB-NN`, `Q-CGT-NN`, `IMP-NN`).
- **Drafting marks.** `[TODO: …]` and `[CHB: …]` notes are drafting marks.

## Keeping memo and code in step

A memo here describes what the code computes. If a memo has to change, the code probably has to change too.
- Edit the LaTeX source, not the Markdown.
- Regenerate the copy with `make memo-md NAME=M003_receiver_calibration OUT=<eigsep_cal>/docs/memos/M003_receiver_calibration`, run from the manuscript.
- Commit the regenerated copy in the same PR as the code change.
