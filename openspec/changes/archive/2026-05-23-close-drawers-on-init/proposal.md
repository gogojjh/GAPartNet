## Why

Storage furniture tests currently allow at least one prismatic drawer asset, such as `45427`, to appear open at simulation initialization. Tabletop manipulation evaluations should start from a deterministic closed-drawer state so opening delta, videos, and success labels describe the policy's action instead of inherited asset DOF offsets.

## What Changes

- Ensure prismatic drawer joints are initialized to their closed limit before manipulation starts.
- Apply the closed-state initialization consistently across StorageFurniture/Table assets without modifying dataset asset files.
- Preserve the existing placement rule where only `27044` uses tabletop support and other table assets remain floor-supported.
- Record enough diagnostics in `result.json` to verify the initial DOF was closed.
- Add regression coverage for assets whose prismatic joints have nonzero default/open qpos, including the observed `45427` case.

## Capabilities

### New Capabilities
- `closed-drawer-initialization`: Covers deterministic closed-state initialization for prismatic drawer-style articulated objects during manipulation tests.

### Modified Capabilities

## Impact

- Affected code:
  - `manipulation/run.py` articulated object DOF initialization and diagnostics.
  - Unit tests around run helpers and initialization behavior.
- Affected validation:
  - Re-run `45427` prismatic initialization/open tests with `3d_dp`.
  - Run existing focused test suite for manipulation helpers.
- Dataset asset files must remain unchanged.
