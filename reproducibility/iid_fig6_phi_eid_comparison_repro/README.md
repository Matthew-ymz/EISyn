# IID Fig. 6 PhiEID comparison reproduction bundle

This bundle contains the code and data needed to reproduce the screenshot figure with panels A-D:

- A: mean firing rate vs global coupling `G`
- B: `Phi^R` curves for full-pair cache, uniform pilot, middle-state rows, and tail-biased rows
- C: whole-system `Phi^EID`
- D: identified `G*` from `Phi^R`

## Exact reproduction

Run from this directory:

```bash
python -m pip install -r requirements.txt
./reproduce.sh
```

Expected result:

```text
reproduced_png_sha256=5a1a8a77450041eda88fe29f9c595d489703b6dd6524ec007e828ff2f59c0c57
canonical_png_sha256=5a1a8a77450041eda88fe29f9c595d489703b6dd6524ec007e828ff2f59c0c57
OK: exact PNG reproduced at .../_reproduced/whole_system_phi_eid_phase_comparison.png
```

The exact path uses the frozen numerical cache:

```text
results/iid_fig6_phi_eid_comparison/whole_system_phi_eid_phase_comparison.npz
```

This is the reliable bit-for-bit path. It regenerates the PNG through the original plotting function and checks the SHA-256 checksum against the bundled canonical figure.

## Files included

```text
scripts/reproduce_iid_fig6_phi_eid_comparison.py
exp/brain/dmf_fig6.py
exp/brain/result_lausanne_fig6/count_00_fig6b_mean_rate.npz
results/iid_fig6_phi_eid_comparison/whole_system_phi_eid_phase_comparison.npz
fig/iid_fig6_phi_eid_comparison/whole_system_phi_eid_phase_comparison.png
docs/log/iid_fig6_phi_eid_comparison.md
docs/reports/Part2_DMF_Phi_EID_Critical_Transition.md
docs/reports/assets/part2_dmf_phi_comparison.png
reference/user_supplied_screenshot.png
```

`fig/.../whole_system_phi_eid_phase_comparison.png` is the canonical script output. `docs/reports/assets/part2_dmf_phi_comparison.png` is the report copy used in the Part 2 report.

## Full recompute path

The original recomputation entry point is also included:

```bash
python scripts/reproduce_iid_fig6_phi_eid_comparison.py \
  --source-results exp/brain/result_lausanne_fig6/count_00_fig6b_mean_rate.npz \
  --results _recomputed/whole_system_phi_eid_phase_comparison.npz \
  --figure _recomputed/whole_system_phi_eid_phase_comparison.png \
  --doc _recomputed/iid_fig6_phi_eid_comparison.md \
  --bootstrap-count 12
```

Use this for audit or sensitivity checks. The frozen-cache route above is the one verified for exact byte-identical output, because full stochastic DMF reruns can vary across numerical environments.

## Current environment used to verify

```text
Python 3.11.8
numpy 1.26.4
matplotlib 3.10.8
scipy 1.11.4
```
