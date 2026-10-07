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

1. **Issue.** Work starts as an issue that sets **goals, constraints and motivation**, not a specification, and says whether the work is a foundation or a leaf (see [Sequencing work](#sequencing-work)):
   - *motivation:* why the work matters, and which design sections it serves;
   - *goals:* what should be true when it's done, stated as outcomes, not as APIs or algorithms;
   - *constraints:* what must hold, such as SQL semantics, determinism, compatibility with code already merged, and the size budget;
   - *done when:* how a reviewer will know, such as which properties the tests establish;
   - the branch it builds on.

   Leave the design of the code (types, function signatures, algorithms) to the implementer, who should judge from the code and the design doc. A prescribed API biases the implementation. If an interface must be shared between parallel pieces of work, agree on it in the issue's discussion or in a small pull request that lands first.

   Milestones have tracking issues.
2. **Branch.** One branch per issue, named `<milestone>/<item>-<topic>`, such as `m1/c1-hash-kernel`.
3. **Draft pull request,** linked to its issue (`Closes #N`), and based on the branch it builds on. Pull requests may stack; when a base merges, GitHub retargets the pull requests on top of it.
4. **Design review by the implementer.** Before marking a pull request ready, its author checks it against the design sections it implements, and says in the description how it adheres, or which RFC proposes the difference. Any agent or person may also review for the design; nobody is a required gate for it.
5. **Ready for review.** Mark the pull request ready once CI is green and the design review is done.
6. **Approval and merge** by a maintainer or code owner ([`.github/CODEOWNERS`](.github/CODEOWNERS)).

## Sequencing work

Concrete code review is where a design's specifics get settled, and it can't all happen in advance. Rebuilding work that was built on code whose design then changed is the most expensive thing we do. These rules keep review early and rework small. They're an experiment; change them as we learn.

### Foundations and leaves

Every pull request is one of two kinds, which its issue states:

- A **foundation** defines something other work depends on: shared types, a trait, an operator's contract, a public API.
- A **leaf** depends on foundations, but nothing depends on it: an implementation behind an agreed interface, tests, benchmarks, docs, a spike.

### The review frontier

Work may build at most **one layer past code a maintainer has reviewed**. Nothing may stack on a foundation until a maintainer has reviewed it and agreed its shape. The foundation doesn't need to be merged, but it does need that agreement. Leaves may proceed in parallel freely.

The depth of unreviewed work is what causes rework, more than the number of open pull requests. As a secondary guard on review load, aim for **at most five pull requests awaiting maintainer review** at once.

### Interfaces first

The first pull request for a new component is its **interface**: public types, signatures and doc comments, with stub bodies (`todo!()` is fine there). It is small, and quick to review and to change. Implementation pull requests follow, once the interface is agreed.

### A tracer bullet before breadth

For each milestone, first build one **thin end-to-end path**: the smallest case that exercises every layer, such as one aggregate over two tables, from detection through execution. A maintainer reviews that whole slice, because design problems show up where the pieces meet. Only then widen it, in parallel.

### While waiting for review

Agents don't build on unreviewed foundations while they wait. Instead, they work on things that don't depend on them:

- test generators and oracles;
- benchmarks against the unmodified engine;
- spikes;
- docs;
- reviewing each other's pull requests against the design.

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

- When several agents work at once in separate git worktrees, give each worktree **its own** Cargo target directory. Cargo assigns the same build hash to a crate in different worktrees, so a shared target directory can run another worktree's build of your crate.
- DataFusion is large: limit parallel build jobs (`CARGO_BUILD_JOBS`) on machines with little memory.
- Run the CI checks before pushing: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`, and `reuse lint`.
