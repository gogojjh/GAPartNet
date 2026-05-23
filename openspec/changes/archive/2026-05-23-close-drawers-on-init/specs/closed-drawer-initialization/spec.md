## ADDED Requirements

### Requirement: Prismatic drawers initialize closed
The manipulation runner SHALL command every prismatic drawer DOF on the articulated object to its closed limit before the manipulation trajectory and video recording begin.

#### Scenario: Multi-drawer asset starts closed
- **WHEN** a StorageFurniture/Table asset such as `45427` contains multiple prismatic drawer joints
- **THEN** each prismatic drawer joint is initialized at its closed limit before the first manipulation frame is evaluated

#### Scenario: Target drawer delta measures manipulation only
- **WHEN** a prismatic target joint is tested
- **THEN** the recorded opening delta is measured from the closed initial DOF state rather than from an inherited open asset state

### Requirement: Dataset assets remain unchanged
The implementation MUST enforce closed initialization at runtime without modifying GAPartNet dataset asset files.

#### Scenario: Runtime-only initialization
- **WHEN** the closed drawer initialization fix is applied
- **THEN** no files under dataset or copied asset directories are edited to encode the closed state

### Requirement: Initialization diagnostics are auditable
The manipulation result SHALL include enough DOF diagnostics to verify whether prismatic drawers were initialized closed.

#### Scenario: Result records initial state
- **WHEN** a prismatic drawer test writes `result.json`
- **THEN** the result includes the target initial DOF and the initialization command or closed reference needed to audit the closed-start condition
