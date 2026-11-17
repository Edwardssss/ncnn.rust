# Documentation map

Each file answers one kind of question. Keep conclusions in these files and dated measurements in
the evidence appendix.

| File | Answers | Update when |
|---|---|---|
| [contracts.md](contracts.md) | What each API tier promises about ownership, lifetimes, unsupported operations and raw escape hatches, and which features are outright non-goals | A public behaviour changes or a scope decision is taken |
| [native-abi.md](native-abi.md) | What the adapter decides where the upstream C API is incomplete, and how far the runtime validation reaches | An adapter decision or a validation campaign changes |
| [layer-coverage.md](layer-coverage.md) | Which layer types have a conformance case, what the coverage table reaches, and which native differences the fixtures avoid | A layer case or a native finding changes |
| [support-matrix.md](support-matrix.md) | Which targets and options are supported, what each CI job proves, how to run ARM64 cross tests | A platform, SDK requirement or CI job changes |
| [troubleshooting.md](troubleshooting.md) | Which symptom maps to which cause and fix: loader path, version pin, link kind, aborts, unavailable devices | A failure mode or a workaround changes |
| [migration.md](migration.md) | What to check when the pinned ncnn release or the crate minor version moves | The pinned release or a breaking change lands |
| [performance.md](performance.md) | How much the wrapper costs against the native C API, method and tables | A copy path, allocation strategy or benchmark campaign changes |
| [advanced-cpu.md](advanced-cpu.md) | Runtime controls, pools, custom layers, tensor operations, readers and images | An advanced adapter gains or changes a contract |
| [model-fixtures.md](model-fixtures.md) | Which trained models are compared, with commands and numeric results | Fixtures, comparisons or their commands change |
| [evidence-history.md](evidence-history.md) | Dated campaign records: machines, counts, defects found | A new validation campaign lands |
| [CHANGELOG.md](../CHANGELOG.md) | User-visible changes per release | A release is prepared |
| [SECURITY.md](../SECURITY.md) | How to report a vulnerability and what the trust boundary is | The reporting route or the scope changes |
| <https://edwardssss.github.io/ncnn.rust/ncnn_rust/> | Generated API reference | Automatically, on every push to the default branch |

The workflows split by what triggers them, which is why a check can be absent from a pull request
without being dropped: `ci.yml` runs per request, `msrv.yml` runs its SDK build only when a manifest
or `rust-toolchain.toml` moves, `coverage.yml` runs weekly, `sdk-artifacts.yml` and
`ncnn-matrix.yml` run on a dispatch, a tag or a schedule, and `release.yml` runs on a version tag.
[Support matrix](support-matrix.md) records what each job proves.

## Writing rules

- State behaviour directly. Describe what the code does, not what it might do.
- Wrap prose at 100 columns. Tables and code blocks keep their own layout, because a wrapped row
  or command is harder to read than a long one.
- Keep validation counts, machine details and log paths in the evidence appendix.
- Prefer a table over prose when the content is a matrix.
- Link the file that owns a topic instead of restating it.
