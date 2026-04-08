# Suggested usage

This skill is general.

To use it with a specific project, add the project plan or spec as a supporting resource, for example:

- `references/project-plan.tex`
- `references/project-proposal.md`
- `references/requirements.pdf`

Then invoke the skill and ask for an implementation plan based on that material.

## Example prompt

Use the ML Implementation Plan Planner skill to create an implementation plan for the project described in `references/plan.tex`.
The system should stay minimal, use DRY/KISS/SOLID pragmatically, use YAML files for experiments, assume correct script/data usage unless otherwise stated, and document those assumptions in the README.
