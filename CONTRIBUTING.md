<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Contributing to einfold

einfold is built in the open by people and AI agents working together. This file describes how work moves from an idea to a merged change, so that anyone can follow it and reproduce it.

## The design comes first

[`docs/design.md`](docs/design.md) is the contract, and [`docs/supplement.md`](docs/supplement.md) holds its details. Code implements the design. If code needs to differ from it, change the design first, in its own PR, or raise it in an issue. Every PR says which design sections it implements.

Principles that reviews check above all (design §4):

- **Never wrong.** A rewrite applies only when it is equivalent under SQL semantics: NULLs, bag semantics, and which groups exist. Exact values stay exact. Missing a speedup is acceptable; changing a result is not.
- **Never add nondeterminism.**
- **The XQL logical model is the contract.** einfold never changes what a query means or the shape of its result.

## From issue to merge

1. **Issue.** Every piece of work starts as an issue with:
   - a spec, with references to the design sections it implements;
   - acceptance criteria;
   - a size budget;
   - the branch it builds on.

   Milestones have tracking issues; for M1, see #2.
2. **Branch.** One branch per issue, named `<milestone>/<item>-<topic>`, such as `m1/c1-hash-kernel`.
3. **Draft PR.** Open the PR as a draft, linked to its issue (`Closes #N`), based on the branch it builds on.
   - PRs may stack: a PR whose base is another PR's branch. When the base merges, GitHub retargets the stacked PR.
   - Keep each PR self-contained and reviewable: about **400 changed lines of non-test code** at most. Tests don't count toward the budget, but must be readable. Split anything larger.
4. **Design review.** 🧭 Claude, the orchestrator, reviews each draft against the design, then marks it ready for review. Review happens in GitHub comments.
5. **Maintainer review and merge.** @alxmrs reviews every PR before it merges. Review agents may also comment.

## Identities

Everyone who posts on GitHub for this project starts each comment, issue, and PR description with their emoji and name, so the record shows who said what.

| | Name | Role |
|---|---|---|
| 🧭 | Claude | Orchestrator: plans the work, keeps the design, reviews every PR before it is marked ready |
| 🔨 | Forge | Implementer for semantics-heavy work |
| 🪛 | Wrench | Implementer for well-specified components |
| 🧪 | Assay | Tests, harnesses, and benchmarks |

Review agents run by @alxmrs introduce themselves with their own emoji.

## Engineering standards

- **Tests prove semantics, not just behavior.** Every rewrite and operator is tested against the plan it replaces, on inputs with NULL values, NULL keys, duplicate coordinate tuples, empty inputs, groups reached only by NULLs, and NaN.
- **Deterministic tests.** Random tests seed their generators and print the seed on failure.
- **No `unsafe`.** It is forbidden workspace-wide.
- **Docs.** Public items have doc comments that say what they mean in SQL terms, and cite the design section they implement.
- **Licensing.** Every file carries SPDX headers, and `reuse lint` passes (the REUSE specification, version 3.3):

  ```
  SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
  SPDX-License-Identifier: Apache-2.0
  ```
- **Pinned engines.** `datafusion` is pinned exactly, in lockstep with [ddx](https://github.com/xqlsystems/ddx). Bump it only together with ddx.

## Local checks

These match CI:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS=-D\ warnings cargo doc --workspace --no-deps
reuse lint
```

## Commit messages and PR descriptions

PRs are squash-merged, and the squash commit's message is the PR's title and description. So:

- the **PR title** is the commit subject, in the imperative mood ("Add the EinFold hash kernel");
- the **PR description** is the commit body: what changed and why, in plain prose, without leftover checklists. Edit it as the PR evolves.

Commits on a branch can be informal, since squashing replaces them. When an AI agent wrote a commit, it adds a `Co-Authored-By:` trailer naming its model; GitHub carries these into the squash commit.
