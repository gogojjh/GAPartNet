## Context

The manipulation runner loads GAPartNet articulated assets and computes opening deltas from the actor DOF state observed during the run. For prismatic drawer tasks, the initial DOF must represent the drawer closed state. If an asset starts with a drawer partially open, the rendered video and delta measurement no longer isolate the manipulation behavior. The observed `45427` StorageFurniture/Table run shows this failure mode.

The change must work without editing dataset asset files and must preserve the already restored object placement behavior: only `27044` is tabletop-supported, while the other tested table assets are placed on the floor support plane.

## Goals / Non-Goals

**Goals:**

- Initialize prismatic target joints to their closed limit before the simulation/video run begins.
- Apply deterministic closed-state initialization to all drawer-style prismatic joints on the articulated object, not only the currently targeted joint, so adjacent drawers do not appear open.
- Keep diagnostics in `result.json` that make the initial DOF state auditable.
- Add regression tests for the closed-limit calculation and DOF initialization command values.

**Non-Goals:**

- Do not modify GAPartNet dataset assets, URDFs, semantics, or mobility metadata.
- Do not change render camera pose, simulation timing, or robot trajectory policy except where the policy consumes the corrected initial articulated state.
- Do not solve low-delta grasp failures in this change; those remain separate manipulation policy issues.

## Decisions

- Use joint limits as the source of truth for closed state. For prismatic drawer axes, the closed command is the lower limit when the joint range is nonnegative. This matches the observed metadata where drawer opening increases from `lower=0.0` toward a positive `upper`.
- Set all prismatic DOFs to closed before stepping the manipulation task. This prevents untargeted drawers from starting open in multi-drawer cabinets like `45427`.
- Keep revolute joint initialization unchanged unless an existing code path already initializes it. The reported defect concerns drawers/prismatic axes, and changing revolute defaults would broaden behavioral risk.
- Store both commanded initial DOF and measured initial DOF in diagnostics where practical. This allows later runs to distinguish “commanded closed but settled open” from “never commanded closed.”
- Implement the behavior in runner-level initialization rather than patching assets. This makes the same rule portable to other programs that import GAPartNet code.

## Risks / Trade-offs

- Some assets may encode the visually closed pose at the upper limit rather than lower. Mitigation: start with the current StorageFurniture/Table convention and add diagnostics so exceptions are visible; handle exceptions with metadata only if evidence appears.
- If the simulator ignores a DOF command until after a settling step, the first measured frame could still show slight numerical offset. Mitigation: compare with a small tolerance and record exact initial values.
- Closing all prismatic joints can change behavior for assets that previously relied on a non-target drawer being open. Mitigation: this is intended for tabletop manipulation evaluation consistency; policy recovery is handled separately.
