<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

How AI agents work on einfold. Everything in [`CONTRIBUTING.md`](CONTRIBUTING.md) applies to agents too; this file adds the structure that makes agent work reviewable and reproducible. People working with agents may find it useful as well.

## Know the design

[`docs/design.md`](docs/design.md) describes what einfold is and why, and [`docs/supplement.md`](docs/supplement.md) holds the details, numbered to match. Read the sections relevant to your task before you start. Code implements the design. If the code needs to differ from it, propose the change as an RFC ([`docs/rfcs/README.md`](docs/rfcs/README.md)) instead of drifting silently.

The design matters for planning, but code must not depend on it: comments and docs in code are self-contained (see [Code comments](#code-comments)).

## From issue to merge

1. **Issue.** Work starts as an issue with:
   - a spec;
   - the design sections it implements;
   - acceptance criteria;
   - a size budget;
   - the branch it builds on.

   Milestones have tracking issues.
2. **Branch.** One branch per issue, named `<milestone>/<item>-<topic>`, such as `m1/c1-hash-kernel`.
3. **Draft pull request,** linked to its issue (`Closes #N`), and based on the branch it builds on. Pull requests may stack; when a base merges, GitHub retargets the pull requests on top of it.
4. **Design review by the implementer.** Before marking a pull request ready, its author checks it against the design sections it implements, and says in the description how it adheres, or which RFC proposes the difference. Any agent or person may also review for the design; nobody is a required gate for it.
5. **Ready for review.** Mark the pull request ready once CI is green and the design review is done.
6. **Approval and merge** by a maintainer or code owner ([`.github/CODEOWNERS`](.github/CODEOWNERS)).

## Size

Aim for about **400 changed lines of non-test code** per pull request. When work is bigger:

- **split it into stacked pull requests,** each coherent and reviewable on its own. This is the default;
- **or open a follow-up issue** for the part that can genuinely wait, and link it from the pull request.

## Identities

Agents start every GitHub comment, issue and pull request description with their emoji and name, so the record shows who said what. Pick an emoji not already in use, and keep it for the life of the agent. The team so far:

| | Name | Role |
|---|---|---|
| 🧭 | Claude | Orchestrator: plans work, writes issue specs, reviews for the design |
| 🔨 | Forge | Implementer for semantics-heavy work |
| 🪛 | Wrench | Implementer for well-specified components |
| 🧪 | Assay | Tests, harnesses and benchmarks |

Review agents run by maintainers introduce themselves with their own emoji.

## Pull request descriptions and commits

Pull requests are squash-merged, and the squash commit's message is the pull request's title and description. So:

- the **title** is the commit subject, in the imperative mood ("Add the EinFold hash kernel");
- the **description** is the commit body: plain prose saying what changed and why. Keep checklists inside HTML comments, and edit the description as the pull request evolves;
- every commit an agent writes carries a `Co-Authored-By:` trailer naming its model. GitHub carries these into the squash commit.

## Code comments

A maintainer must be able to understand a file without the design doc. So:

- **Never cite the design doc from code** (no "design §8.3"). Instead, explain the concept where it is used: what a partial aggregate is, why NULL keys matter, why an order is fixed.
- **Say what the code means in SQL terms** where that helps: which query it is equivalent to, and how NULLs and duplicates behave.
- **Keep comments true.** Update them with the code.

## Tests

Follow the two kinds described in [`CONTRIBUTING.md`](CONTRIBUTING.md#tests). A maintainer reviews property-based, fuzz and simulation tests closely, and only runs hand tests. So:

- put effort into tests that state invariants and search for violations;
- say in the pull request description which invariants the tests check;
- keep hand tests readable, and don't count on reviewers checking each one.

## Bugs in dependencies

File them **in this repository**, never directly upstream, with the `upstream` label. Include a minimal reproduction, and a test in einfold that pins the current behavior and fails once it is fixed. A maintainer decides whether to report it upstream.

## Building locally

- When several agents work at once in separate git worktrees, give each worktree **its own** Cargo target directory. Cargo assigns the same build hash to a crate in different worktrees, so a shared target directory can run another worktree's build of your crate.
- DataFusion is large: limit parallel build jobs (`CARGO_BUILD_JOBS`) on machines with little memory.
- Run the CI checks before pushing: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`, and `reuse lint`.
