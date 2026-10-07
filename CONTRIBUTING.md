<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Contributing to einfold

Thank you for being here! einfold is built in the open, and contributions of every size are welcome, including your first one. This guide shows you how to get set up and how changes get in.

By contributing, you agree to follow the XQL Systems [code of conduct](https://github.com/xqlsystems/.github/blob/main/CODE_OF_CONDUCT.md).

If you are an AI agent, or you work with agents, also read [`AGENTS.md`](AGENTS.md), which describes the more structured process agents follow.

## Ways to contribute

You don't need to write Rust to help:

- **File an issue.** A bug, a confusing error, an idea, or a question is a contribution.
- **Improve the docs.** Fixing a typo or clarifying a sentence is a real contribution that helps everyone who reads it after you.
- **Write tests.** Tests that find bugs are among the most valuable things you can add (see [Tests](#tests)).
- **Report bugs in our dependencies** (see [Bugs in dependencies](#bugs-in-dependencies)).
- **Propose a design change** through an RFC (see [Changing the design](#changing-the-design)).
- **Send a pull request** with a fix or a feature.

If you're not sure where to start, look for issues labeled [`good first issue`](https://github.com/xqlsystems/einfold/labels/good%20first%20issue), or open an issue and ask.

## Getting set up

1. **Install Rust** with [rustup](https://rustup.rs). einfold needs Rust 1.88 or newer:

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone the repository.** We recommend SSH, which lets you push without typing a password once you've [added an SSH key to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account):

   ```sh
   git clone git@github.com:xqlsystems/einfold.git
   cd einfold
   ```

   Or clone over HTTPS, which needs no setup:

   ```sh
   git clone https://github.com/xqlsystems/einfold.git
   ```

3. **Build and run the tests:**

   ```sh
   cargo test --workspace
   ```

   The first build compiles Apache DataFusion, the query engine einfold plugs into. Expect it to take several minutes and a few gigabytes of disk. Later builds are much faster.

4. **Optional: install `reuse`,** the tool that checks every file's license header. CI runs it for you, so this is only for checking locally. We use [uv](https://docs.astral.sh/uv/), the Python package manager, for Python tools:

   ```sh
   uv tool install reuse
   reuse lint
   ```

Before opening a pull request, it helps to run the same checks CI runs:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Making a change

1. **Fork** the repository on GitHub, and create a branch for your change.
2. **Keep it small.** A pull request that does one thing is much easier to review than one that does several. If your change grows large, split it into several pull requests, each building on the one before. It's fine to open a follow-up issue for part of the work that can wait.
3. **Open a pull request** against `main`, and describe what you changed and why. Draft pull requests are welcome if you'd like early feedback.
4. **Review.** A maintainer or code owner (see [`.github/CODEOWNERS`](.github/CODEOWNERS)) reviews and approves every pull request. Pull requests are squash-merged, so your branch's commit history doesn't need to be tidy.

Don't worry about getting everything right the first time. Reviewers will help.

## What reviewers look for

einfold rewrites other engines' query plans, so above all it must never change a query's answer. Reviewers check that:

- **Results are unchanged,** including the corners of SQL: NULL values and keys, duplicate rows, groups that no row reaches, and NaN. When einfold can't be sure a rewrite is safe, it leaves the plan alone.
- **Exact values stay exact,** such as integers, `DECIMAL`, `COUNT`, `MIN` and `MAX`.
- **Nothing new depends on timing.** einfold never makes a query less repeatable than it was.
- **The result's shape stays the same,** with the same columns and types. This keeps einfold's output portable across SQL engines.

## Writing code

- **Code explains itself.** Comments and docs must introduce every concept the code relies on, so a maintainer can understand a file without reading the design doc. Explain what something means in SQL terms where that helps.
- **No `unsafe`.** It is forbidden across the workspace.
- **Document public items** with doc comments.
- **License headers.** Every file starts with SPDX tags. `reuse annotate` can add them for you:

  ```sh
  reuse annotate --copyright="Alexander Merose <al@merose.com> & einfold Authors" \
    --license=Apache-2.0 --year=2026 path/to/file
  ```

- **Pinned engines.** The `datafusion` dependency is pinned exactly, in lockstep with [ddx](https://github.com/xqlsystems/ddx), a sister project that uses einfold. Please don't bump it in an unrelated change.

## Tests

einfold distinguishes two kinds of test, following Dan Luu's ["AI coding"](https://danluu.com/ai-coding/) notes:

- **Hand tests** are examples written one at a time: this input gives that output. They help build and check the code, and they're welcome. They should be readable. Reviewers run them, but don't scrutinize each one.
- **Tests,** in the stronger sense, check properties over many inputs: property-based tests, fuzzing, and simulation tests, such as einfold's equivalence harness (`crates/einfold-testkit`), which compares rewritten queries against SQL itself on random inputs. These get careful review, because they are what find bugs. Good ones state an invariant and hunt for inputs that break it.

Random tests must be reproducible: seed the generator, and print the seed when a test fails. For why simulation testing finds bugs other tests miss, see the talk ["The Rocket Science of Simulation Testing"](https://www.hytradboi.com/2025/c222d11a-6f4d-4211-a243-f5b7fafc8d79-rocket-science-of-simulation-testing) (HYTRADBOI 2025).

## Bugs in dependencies

If you find a bug in a dependency, such as DataFusion, you're welcome to report it to that project directly. If it affects einfold, please also open an issue here with the `upstream` label and a link, so we can track it.

## Changing the design

[`docs/design.md`](docs/design.md) describes what einfold is and why, and [`docs/supplement.md`](docs/supplement.md) holds the details. To change the design, write a short RFC (request for comments); see [`docs/rfcs/README.md`](docs/rfcs/README.md). An RFC is easier to discuss than a diff of the design doc. Once it's accepted, the design doc is updated to match.

## License

By contributing, you agree that your contribution is licensed under the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt).
