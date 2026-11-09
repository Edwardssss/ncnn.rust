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
- `cargo clippy --workspace --all-targets -- -D warnings` and the same command with
  `--features vulkan`, which covers the feature-gated test code
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
words a report uses. `.github/issue-labeler.yml` holds one regular expression per label: `bug`,
`enhancement`, `documentation`, `question`, `performance`, `build`, `vulkan`, the four
`platform:*` labels, `needs-repro` and `needs-info`. The action matches the title and the body and
joins several entries for one label with AND, so alternatives belong inside a single
case-insensitive expression of the `/pattern/i` form. Matching labels stay in place until a
maintainer removes them.

Applying or removing a label requires the triage role or higher. The repository grants write
access to `Edwardssss` alone, so external contributors cannot label their own issues or pull
requests.

## Formatting

| Language | Tool | Command |
|---|---|---|
| Rust | rustfmt (`rustfmt.toml`) | `cargo fmt --all` |
| C++ adapters | clang-format (`.clang-format`) | `clang-format -i crates/ncnn-sys/native/*.cpp crates/ncnn-sys/native/*.h` |
| Markdown, JSON, YAML | EditorConfig (`.editorconfig`) | Two-space indent, LF, final newline |

`rust-toolchain.toml` pins the stable channel with `rustfmt` and `clippy`; the MSRV job pins
1.85.1 separately, so code that relies on a newer language feature breaks that job even when the
local stable toolchain accepts it. `.gitattributes` normalises text files to LF and marks native
binaries. Do not reflow unrelated code in a functional change, and do not disable a lint without
documenting the reason next to it.

## Documentation

`README.md` and `README_zh.md` are the same document in two languages. A change to one lands in the
same pull request as the matching change to the other, so the two never describe different APIs.

Prose and code comments are written in English as declarative statements about what the code does.
Text that explains why a reader might be surprised, or that apologises for a limitation, belongs in
`docs/` next to the contract it qualifies, not in a comment.

`docs/README.md` lists every page. A new page is added to that map, to the documentation list in
both READMEs and to `CHANGELOG.md` in the same pull request.

## Dependencies

Two axes move independently, and each lands as its own pull request so a failure names one change.

- **Crate dependencies.** `.github/workflows/dependabot-auto-merge.yml` approves patch and minor
  Dependabot pull requests and arms auto-merge; major updates wait for a maintainer. `Cargo.lock` is
  committed and updated with the change that requires it, so a lock-only pull request states which
  dependency forced the update. The `ncnn-rust-sys` dependency of `ncnn-rust` is a path dependency
  carrying the workspace version: it never resolves from the registry inside this workspace, and the
  release workflow publishes the two packages in order because the published metadata does.
- **The ncnn SDK.** `ci/ncnn-versions.json` names the pinned release and the neighbours the monthly
  matrix builds. Moving the pin follows `docs/migration.md` and never shares a pull request with a
  crate dependency update. Features do not select the SDK: `NCNN_INSTALL_DIR` does, and every job
  builds it through `tools/build-ncnn-sdk.py` so the patches in `tools/patches/` always apply.
- **The toolchain floor.** `rust-version` moves separately from both. Raising it drops a runtime
  rather than adapting to one, so it needs its own pull request with the supported-runtime statement
  in `docs/support-matrix.md`.

A dependency that changes inference numbers, copy paths or native behaviour needs the evidence of a
code change: the conformance suite, the Vulkan contracts and the trained-model comparison for the
affected path. A dependency that only affects tooling needs the job that uses it.

## Versions and releases

`ncnn-rust` and `ncnn-rust-sys` are published together and always carry the same version.

Preparing a release stays manual:

1. Confirm `repository`, `homepage` and `documentation` in both `crates/*/Cargo.toml` manifests name
   this repository; `cargo publish` warns and the crates.io page stays empty when they are missing.
2. Set `version` in the workspace manifest and in the `ncnn-rust-sys` dependency entry, run
   `python tools/check-versions.py`, and let the `Semver checks` job report whether the new version
   allows the changes it carries.
3. Move the `Unreleased` entries of `CHANGELOG.md` under the new version heading, add the release
   date and a fresh empty `Unreleased` section. The release body comes from that section, so a
   version without one fails the release workflow.
4. Tag the commit and push the tag.

Pushing a `v*` tag starts `.github/workflows/release.yml`:

1. It refuses a tag whose name differs from the workspace version.
2. It publishes `ncnn-rust-sys`, waits for its crates.io index entry, then publishes `ncnn-rust`. The
   wait is what orders them, because the second package resolves the first from the registry. A
   package that already holds that version is skipped, so re-running a tag is safe.
3. It creates the GitHub release from the matching `CHANGELOG.md` section and updates the release
   when one already exists.

The workflow needs `CARGO_REGISTRY_TOKEN` in the `release` environment. `.github/release.yml` groups
generated notes by label for manual `gh release create --generate-notes` runs; the workflow itself
uses the changelog section.

The ncnn SDK version is a separate axis. `ci/ncnn-versions.json` records the pinned release, and
`crates/ncnn-sys/build.rs` rejects an unlisted SDK version unless `NCNN_ALLOW_UNTESTED_VERSION=1`
is set.
