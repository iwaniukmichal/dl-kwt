# Suggested usage

This skill is general.

To use it with a specific project, ensure the project plan or spec is available as a supporting resource, for example:

- `references/plan.tex`
- `references/project-plan.md`
- `references/spec.pdf`

The skill also requires a `README.md` in the repository root with a section describing end-to-end experiment instructions (e.g. `## End-to-end experiments for the project plan`).

Then invoke the skill and ask it to validate that the README instructions fully cover the project plan.

## Example prompt

Use the Plan Execution Validation skill to verify that the README instructions in this repository fully cover the project plan in `references/plan.tex`. Fix any missing steps, unclear instructions, or implementation gaps.

## What it does

1. Reads the project plan and extracts all required experiments, phases, deliverables, and metrics.
2. Reads the README's end-to-end experiment section and inventories every instruction.
3. Cross-references plan requirements against README instructions.
4. Validates that every referenced config file, CLI entrypoint, and output path actually exists.
5. Produces a validation report with gap analysis.
6. Applies fixes to the README and/or implementation to close gaps.
7. Confirms the README now fully covers the plan.
