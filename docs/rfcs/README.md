<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# RFCs: changing einfold's design

[`docs/design.md`](../design.md) and [`docs/supplement.md`](../supplement.md) describe what einfold is and why. To change them, write an RFC (a request for comments). An RFC states one proposed change and its reasons, and is easier to discuss than a diff of a long document.

## When to write one

Write an RFC for changes to:

- principles, goals or non-goals;
- the architecture: components, output forms, deployment modes;
- decisions recorded in the design, such as how facts travel or how determinism works;
- anything an implementation found it had to do differently from the design.

Fixes to typos, clarifications, and updates recording spike results don't need an RFC: edit the design directly in a pull request.

## How

1. Copy [`0000-template.md`](0000-template.md) to `NNNN-short-title.md`, using the next free number.
2. Open a pull request titled `RFC NNNN: <title>`, containing only the RFC. Discussion happens on the pull request.
3. A maintainer or code owner accepts, rejects or defers it. Accepted RFCs are merged with status `Accepted`; rejected and deferred ones may be merged with that status too, so the reasoning is kept.
4. Once accepted, the design doc is updated to match, in the same pull request or a follow-up that cites the RFC.
