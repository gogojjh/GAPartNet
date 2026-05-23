## 1. Investigation

- [x] 1.1 Inspect the current articulated object DOF initialization path in `manipulation/run.py`.
- [x] 1.2 Confirm the observed `45427` open-start behavior from existing outputs or a focused diagnostic run.
- [x] 1.3 Identify how prismatic joint limits and DOF indices are exposed by the loaded articulation.

## 2. Regression Tests

- [x] 2.1 Add a helper-level test for selecting closed prismatic DOF command values from joint limits.
- [x] 2.2 Add a helper-level test that all prismatic DOFs are included in the closed initialization command.
- [x] 2.3 Verify the new tests fail before production code changes.

## 3. Implementation

- [x] 3.1 Implement runtime closed-state initialization for prismatic drawer DOFs without modifying asset files.
- [x] 3.2 Preserve existing tabletop/floor support placement behavior.
- [x] 3.3 Add result diagnostics for target initial DOF and closed initialization reference.

## 4. Verification

- [x] 4.1 Run focused unit tests in the `3d_dp` environment.
- [x] 4.2 Re-run a focused `45427` prismatic test and inspect the initial DOF diagnostics.
- [x] 4.3 Confirm `git diff -- manipulation/assets` remains empty.
