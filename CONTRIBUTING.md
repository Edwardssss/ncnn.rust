# Contributing

## Commits

Commit subjects follow Conventional Commits:

```
<type>(<scope>): <imperative summary>
```

| Type | Use |
|---|---|
| `feat` | New public API or behaviour |
| `fix` | Defect repair |
| `perf` | Measured performance change |
| `refactor` | Internal restructuring without behaviour change |
| `docs` | Documentation only |
| `test` | Test-only change |
| `build` | Cargo, linker or adapter build changes |
| `ci` | Workflow or tooling change |
| `chore` | Maintenance without behaviour change |

Scopes: `net`, `mat`, `gpu`, `options`, `image`, `sys`, `bench`, `ci`, `docs`.

Rules:

- Subject is 72 characters or fewer, imperative, and has no trailing period.
- The body explains what changed and why. Measurements, hardware and commands belong here.
- Reference the affected platform when behaviour is platform-specific.
- Use `Co-authored-by:` trailers when a commit has several authors.
- One logical change per commit. Reformatting and behaviour changes stay separate.

## Branches

`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`. Branches merge through a pull
request; direct pushes to `main` are reserved for release tags.

## Pull requests

A pull request states the motivation, the public API impact and the validation performed.
Include the exact commands and their outcomes. Changes to inference behaviour, ownership
contracts or the native adapter require:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --features vulkan -- -D warnings`
- `cargo test --workspace --features vulkan` plus the CPU-only workspace test
- An SDK built by `tools/build-ncnn-sdk.py`, which applies `tools/patches/`; rerun the affected
  contract and comparison evidence after changing a patch
- A focused comparison run when timings or copy paths change
- An updated document when a contract in `docs/` changes

New operators or parameters belong in `tools/patches/`-aware comparison cases or in the in-repo
conformance suite under `crates/ncnn/tests/conformance/`. The conformance suite needs no
downloaded fixtures; add a family entry there when a layer gains a validated reference loop.

The crate compiles with `#![deny(missing_docs)]`. Every public item carries a doc comment, and
documentation states behavior in declarative sentences instead of hedging or restating the code.

`.coderabbit.yaml` holds the automated review configuration: it skips vendored sources, generated
fixtures and lock files, runs Clippy plus the workflow and Markdown linters, and gives the Rust,
adapter, build-script, workflow and documentation paths their own review instructions. Delete the
file when the repository does not use CodeRabbit.

## Merging

The `main` ruleset requires one approving review, a resolved review thread per finding, and the
`test (ubuntu-latest)` plus `MSRV 1.85.1` checks. Squash is the only merge method, and only
repository administrators can bypass the ruleset. External contributions therefore need a
maintainer approval before they land.

Dependency updates merge without a maintainer: `.github/workflows/dependabot-auto-merge.yml`
approves patch and minor Dependabot pull requests and arms auto-merge, which completes once the
required checks pass. Major updates wait for a maintainer. The Dependabot app holds a
`pull_request` bypass entry so it can complete its own merges, and it holds no push rights.

The `v*` tag ruleset blocks deleting or moving a release tag. Repository settings offer squash as
the only merge method, so the merge buttons match the ruleset, and the Dependabot alerts plus
secret scanning run on every push.

## Labels

`.github/labeler.yml` maps changed paths to labels, and `.github/workflows/labeler.yml` applies
them through the `pull_request_target` event, which supplies the write token that fork pull
requests do not receive under the plain `pull_request` event. The workflow never checks out pull
request code, so untrusted content never runs with that token. `sync-labels` stays off, which
keeps maintainer labels such as `bug` in place after a later push.

`.github/release.yml` groups generated release notes by the same labels, so `gh release create
--generate-notes` produces the category list from the labels a pull request carries.

Issues carry free-form text, so `.github/workflows/issue-labeler.yml` classifies them from the
words a report uses. `.github/issue-labeler.yml` holds one pattern list per label: `bug`,
`enhancement`, `documentation`, `question`, `performance`, `build`, `vulkan`, the four
`platform:*` labels, `needs-repro` and `needs-info`. A label applies when any pattern matches the
title or the body, and matching labels stay in place until a maintainer removes them.

Applying or removing a label requires the triage role or higher. The repository grants write
access to `Edwardssss` alone, so external contributors cannot label their own issues or pull
requests.

## Formatting

| Language | Tool | Command |
|---|---|---|
| Rust | rustfmt (`rustfmt.toml`) | `cargo fmt --all` |
| C++ adapters | clang-format (`.clang-format`) | `clang-format -i crates/ncnn-sys/native/*.cpp crates/ncnn-sys/native/*.h` |
| Markdown, JSON, YAML | EditorConfig (`.editorconfig`) | Two-space indent, LF, final newline |

`rust-toolchain.toml` pins the stable channel with `rustfmt` and `clippy`. `.gitattributes`
normalises text files to LF and marks native binaries. Do not reflow unrelated code in a
functional change, and do not disable a lint without documenting the reason next to it.

## Versions and releases

`ncnn-rust` and `ncnn-rust-sys` are published together and always carry the same version. A
release performs these steps:

1. Confirm `repository`, `homepage` and `documentation` in both `crates/*/Cargo.toml` manifests name
   this repository; `cargo publish` warns and the crates.io page stays empty when they are missing.
2. Set `version` in both manifests and in the workspace dependency entry, then run
   `python tools/check-versions.py`, which compares every `ncnn-rust*` package version with
   `cargo metadata`.
3. Run `cargo publish -p ncnn-rust-sys --dry-run` and the matching `ncnn-rust` dry run, which CI
   also performs on every pull request.
4. `cargo publish -p ncnn-rust-sys`, wait for the index update, then `cargo publish -p ncnn-rust`.
5. Tag the commit and create the GitHub release with generated notes, so the tag records the
   published source.

No changelog file is maintained. Release notes come from GitHub Releases, which groups the
Conventional Commit subjects recorded between two tags, so the commit rules above carry the
release history.

The ncnn SDK version is a separate axis. `ci/ncnn-versions.json` records the pinned release, and
`crates/ncnn-sys/build.rs` rejects an unlisted SDK version unless `NCNN_ALLOW_UNTESTED_VERSION=1`
is set.
