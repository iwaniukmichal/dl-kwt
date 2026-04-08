---
name: ml-implementation-planner
description: Create a concrete implementation plan for a machine learning or deep learning experiment system from a project plan, proposal, or research spec.
---

# ML Implementation Planner

## Goal

Turn a project plan, proposal, thesis outline, experiment brief, or similar specification into a concrete implementation plan for an ML/DL system.

The output must be an **implementation plan** in markdown format, not code. Create `implementation-plan.md` in docs/ folder.

The plan should optimize for:

- minimalism
- maintainability
- reproducibility
- easy experiment execution
- clear scope boundaries
- pragmatic software design

## Input expectations

The user may provide one or more of the following:

- a project plan file (LaTeX, Markdown, PDF, DOCX, text)
- a short project description in chat
- dataset notes
- model/experiment requirements
- constraints on tooling or style

If a plan/spec file is provided, treat it as the main source of truth.
Ask at most 1–3 focused follow-up questions, if you see some blockers or unclarities. If not blocked, proceed with reasonable assumptions and state them clearly.

## Design philosophy

You are planning a **research/experimentation system**, not a generic ML platform.

Follow these principles:

1. **Build only what the project needs.**
   Do not design a general-purpose framework unless the requirements genuinely demand it.

2. **Prefer KISS over abstract purity.**
   Use DRY and SOLID pragmatically, not aggressively. Avoid premature abstraction.

3. **Prefer explicitness over magic.**
   Keep configuration, experiment execution, outputs, and control flow easy to understand.

4. **Prefer composition over inheritance.**
   Small, focused modules are better than deep class hierarchies.

5. **Make experiments easy to run and inspect.**
   A user should be able to understand how an experiment works by reading its config and the few modules it touches.

6. **Use YAML for experiment configuration whenever it fits the project.**
   Prefer one YAML file per runnable experiment. Only introduce shared defaults/includes if they clearly reduce harmful duplication without making the system harder to reason about.

7. **Assume correct usage unless the spec says otherwise.**
   Do not plan for many edge cases or broad defensive programming. Add only necessary validation/error handling, such as:
   - missing config file
   - missing dataset/checkpoint/path
   - impossible or contradictory config values when they are easy to detect

8. **Reproducibility is a first-class requirement.**
   Include seed management, deterministic settings when reasonable, config snapshots, and structured output directories.

9. **Keep the toolchain lightweight.**
   Do not recommend heavyweight orchestration or configuration frameworks unless there is a strong project-specific reason.

## What to extract from the project spec

Read the provided plan/spec carefully and extract:

- project goal and research questions
- dataset(s), splits, and preprocessing requirements
- model families / architectures / backbones
- experiment phases or milestones
- training modes (for example supervised, self-supervised, few-shot, transfer learning, evaluation-only)
- hyperparameter search requirements
- augmentation or preprocessing studies
- evaluation metrics
- visualization / analysis requirements
- reproducibility requirements
- compute or storage constraints
- expected outputs and artifacts
- optional extensions vs required core functionality

## Planning rules

When producing the implementation plan:

- design the smallest system that can support the required experiments cleanly
- do not invent features that are not justified by the spec
- do not propose a single unified training engine if separate loops would be simpler and clearer
- share components where reuse is real and stable
- allow deliberate duplication when abstraction would make the code harder to understand
- keep execution flow simple: config -> build components -> run experiment -> save outputs -> aggregate results
- tie every major module to a real need from the spec

## Default stack guidance

Unless the project constraints say otherwise, you may assume a lightweight Python-based research stack.
For deep learning projects, a typical default may include only what is needed, for example:

- PyTorch / (Lightning if it fits the project)
- torchvision / timm when appropriate
- scikit-learn for classical baselines or analysis utilities
- matplotlib for plots
- PyYAML or equivalent for config loading

## Required output structure

Return the `docs/implementation-plan.md` using this structure exactly:

### 1. System intent

A short paragraph describing what the system exists to do.

### 2. Assumptions and engineering philosophy

State the assumptions you are making and how DRY, KISS, SOLID, minimalism, and limited error handling should be interpreted for this project.

### 3. Derived requirements from the spec

Summarize the concrete capabilities the codebase must support.

### 4. Proposed tech stack

List only tools/libraries that are genuinely needed and briefly justify each one.

### 5. Repository structure

Provide a concrete, minimal repository tree and briefly explain each top-level directory and file. Prefer `src/` for the system implementation, `outputs/` directory for output files of experiments, keep `README.md` in the repository root, and include only folders/resources that are justified by the project scope and common best practices.

### 6. Core modules and responsibilities

For each core module, explain:

- responsibility
- why it exists
- what it should not do

### 7. YAML configuration strategy

Explain:

- how experiment YAML files are organized
- what fields they contain
- how seeds, paths, model choice, training mode, augmentations, and outputs are represented
- how to keep configs readable
- when shared defaults are acceptable and when explicit duplication is better

### 8. Execution flows

Describe the minimal execution flow for each required experiment type.
Examples may include:

- standard training
- evaluation-only run
- hyperparameter search batch
- few-shot / episodic run
- embedding analysis job
- ensemble evaluation
  Only include flows that are actually relevant to the spec.

### 9. Phase-by-phase implementation roadmap

Break implementation into ordered milestones aligned with the project phases or work packages.
For each milestone include:

- deliverables
- dependencies
- validation criteria

### 10. Reuse vs deliberate duplication

State clearly:

- what should be shared
- what should intentionally remain separate for clarity and simplicity

### 11. Logging, outputs, and artifact organization

Cover:

- per-run outputs
- config snapshots
- checkpoints
- metrics files
- plots/tables
- seed aggregation outputs
- final comparison artifacts

### 12. README requirements

Specify exactly what the README should document, including:

- general repository description and purpose
- installation and quick start
- repository structure and a high-level description of the core modules
- how to use the system: CLI, configs, data formats, and input expectations
- output directory conventions
- how to run end-to-end experiments that fulfill the project plan

### 13. Risks / open questions

List only real uncertainties that could change the implementation plan.
If there are no major blockers, keep this section short.
