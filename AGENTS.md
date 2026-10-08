<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

How AI agents work on einfold. Everything in [`CONTRIBUTING.md`](CONTRIBUTING.md) applies to agents too; this file adds the structure that makes agent work reviewable and reproducible. People working with agents may find it useful as well.

## Know the design

[`docs/design.md`](docs/design.md) describes what einfold is and why, and [`docs/supplement.md`](docs/supplement.md) holds the details, numbered to match. [`docs/lessons.md`](docs/lessons.md) records what review and testing have taught us that the design doesn't say. Read the sections relevant to your task before you start. Code implements the design. If the code needs to differ from it, change the design doc in the same or an earlier pull request, instead of drifting silently.

The design matters for planning, but code must not depend on it: comments and docs in code are self-contained (see [Code comments](#code-comments)).

## From request to merge

1. **Start** from a maintainer's request, a milestone in the design, or an issue. An issue states goals, constraints and motivation, not an API: the design of the code is the implementer's to judge.
2. **Branch** per pull request, named `<milestone>/<topic>`, such as `m1/hash-kernel`.
3. **Ready for review** once CI is green.
4. **Approval and merge** by a maintainer or code owner ([`.github/CODEOWNERS`](.github/CODEOWNERS)).

## Size

Aim for about **400 changed lines of non-test code** per pull request. When work is bigger:

- **split it into stacked pull requests,** each coherent and reviewable on its own. This is the default;
- **or open a follow-up issue** for the part that can genuinely wait, and link it from the pull request.

## Identities

Agents start every GitHub comment, issue and pull request description with an emoji and a name, so the record shows who said what. Pick an emoji not already in use, and keep it for the life of the agent.

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

## Introduce every reference

In code, docs, issues and pull requests, introduce every project, paper, tool or term the first time it appears: say what it is in a few words, and link to it. A reader should never have to look up a name to follow the text. For example, write "ddx, an XQL Systems project for automatic differentiation of SQL queries", not just "ddx".

## Tests

Follow the two kinds described in [`CONTRIBUTING.md`](CONTRIBUTING.md#tests). A maintainer reviews property-based, fuzz and simulation tests closely, and only runs hand tests. So:

- put effort into tests that state invariants and search for violations;
- say in the pull request description which invariants the tests check;
- keep hand tests readable, and don't count on reviewers checking each one.

## Bugs in dependencies

Unlike people (see [`CONTRIBUTING.md`](CONTRIBUTING.md)), agents file them **in this repository**, never directly upstream, with the `upstream` label. This spares other projects' maintainers from duplicate or mistaken reports. Include a minimal reproduction, and a test in einfold that pins the current behavior and fails once it is fixed. A maintainer decides whether to report it upstream.

## Building locally

- When working in several git worktrees at once, give each worktree **its own** Cargo target directory. Cargo assigns the same build hash to a crate in different worktrees, so a shared target directory can run another worktree's build of your crate.
- DataFusion is large: limit parallel build jobs (`CARGO_BUILD_JOBS`) on machines with little memory.
- Run the CI checks before pushing: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`, and `reuse lint`.
