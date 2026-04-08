---
name: plan-execution-validation
description: Read a project plan from references, find the end-to-end experiment instructions in the README, and verify that the README instructions fully cover the project plan — fixing README or implementation gaps as needed.
---

# Plan Execution Validation

## Goal

Verify that the project plan can be **fully executed** by following the instructions in the repository README. The skill performs a structured cross-reference between:

1. the **project plan** (the source of truth for what must be done), and
2. the **README end-to-end experiment instructions** (the intended step-by-step guide for fulfilling the plan).

When gaps are found:
- if instructions are missing or unclear → update the README,
- if the implementation does not support a required plan element → update the system code/configs,
- if both are needed → fix both.

The **main deliverable** is a validated README whose end-to-end instructions, when followed exactly, produce all results the project plan demands.


## Required inputs

The repository should contain or provide:

- a project plan file in `references/` or `docs/` (LaTeX, Markdown, PDF, or similar) or other location that user specify.
- a `README.md` with a section describing how to run end-to-end experiments (e.g. `## End-to-end experiments` or similar),
- the implemented codebase (source code, CLI entrypoints, configs),

## Source-of-truth priority

Use sources in this order:

1. explicit user instructions,
2. the **project plan** (defines what must be achieved),
3. the **README end-to-end experiments section** (defines how a user would execute it),
4. the **actual codebase** (configs, CLI entrypoints, modules).

The project plan is the ultimate authority on **what** the project must accomplish.
The README is the authority on **how** a user would accomplish it.
The codebase is the authority on **what is currently possible**.

## Core rules

1. **The project plan defines completeness.**
   Every experiment, phase, deliverable, metric, and analysis described in the project plan must be achievable by following the README instructions.

2. **The README must be self-contained and executable.**
   A reader who follows the README end-to-end section step by step — without reading any additional plans — must produce all required results.

3. **Prefer README fixes over implementation changes.**
   If the implementation already supports a plan requirement but the README fails to document it, fix the README only. Change implementation only when the codebase genuinely cannot support a plan requirement.

4. **Preserve existing correct instructions.**
   Do not rewrite README sections that are already correct and complete. Only modify or add what is needed.

5. **Be explicit about manual steps.**
   If the workflow requires manual intervention, the README must document exactly what to inspect, what to copy, and where to paste it etc.


## Execution workflow

### Phase 1: Read and understand the project plan

Read the project plan file and extract a structured list of:

- project phases / work packages / milestones
- experiments required per phase
- models / architectures to be evaluated
- datasets and data regimes
- hyperparameter search requirements
- analysis tasks (embedding analysis, clustering, visualizations)
- ensemble or combination evaluations
- evaluation metrics expected
- deliverables and expected outputs per phase
- any ordering constraints or dependencies between phases
- reproducibility requirements (seeds, aggregation)
- any other thing that has to be accomplished

Produce a **plan requirements checklist** — a flat list of concrete items that the project must accomplish.

### Phase 2: Read and parse the README experiments section

Read `README.md` and locate the end-to-end experiment section. 
If no such section exists, flag this as a critical gap.

From the README section, extract:

- numbered steps in intended execution order
- CLI commands
- manual actions described between steps
- inter-phase dependencies
- aggregation / summary steps
- any notes or caveats

Produce a **README instruction inventory** — a structured list of what the README tells a user to do.

### Phase 3: Validate against the actual codebase

For every CLI command in the README: Verify that the CLI entry point module exists and correctly implements what is described in the README.

For every manual action in the README: verify that the referenced files and directories exist at that point in the workflow and that they correspond to the correct action.

### Phase 4: Produce a validation report

For each item in the plan requirements checklist, determine:

| Status                | Meaning                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------- |
| ✅ Covered            | The README has a clear, executable instruction that fulfills this plan requirement      |
| ⚠️ Partially covered  | The README mentions it but the instruction is incomplete, ambiguous, or missing details |
| ❌ Missing            | The plan requires it but the README has no corresponding instruction                    |
| 🔧 Implementation gap | The plan requires it but the codebase cannot currently support it  

For each gap or issue, describe:

- what the plan requires,
- what the README currently says (or doesn't say),
- what the codebase supports (or doesn't),
- the recommended fix (README change, implementation change, or both),
- severity (critical = blocks plan completion, minor = unclear but executable, cosmetic = could be clearer).

### Phase 5: Apply fixes

1. **README fixes first.** Edit the README to add missing steps, fix commands, clarify manual actions, and correct references. Follow these sub-rules:
   - maintain the existing README style and formatting,
   - keep the same section hierarchy,
   - add new steps in the correct execution order,
   - preserve existing correct content verbatim,

2. **Implementation fixes second.** If the codebase lacks support for a plan requirement:
   - create missing files,
   - add missing CLI entrypoint capabilities if needed,
   - add minimal module changes only when strictly necessary,

3. **Verify fixes.** After applying changes:
   - re-read the modified README and confirm every plan requirement now has a corresponding instruction,
   - confirm every referenced config/module exists,
   - if the repository has tests, run them to confirm nothing is broken.

## Required response pattern

When using this skill, structure the work around this pattern:

### 1. Project plan understanding

Summarize the project plan's phases, experiments, and deliverables.

### 2. README instruction inventory

Summarize what the README currently instructs a user to do.

### 3. Cross-reference validation

Present the plan requirements checklist with coverage status for each item.

### 4. Gaps and issues

Detail each gap or issue found, with recommended fix and severity.

### 5. Proposed changes

List planned README, implementation, and config changes.

### 6. Applied changes

Describe every change made, with file paths and brief rationale.

### 7. Final validation state

State whether the README now fully covers the project plan, and list any remaining blockers or caveats.