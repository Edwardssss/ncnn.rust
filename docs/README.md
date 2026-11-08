# Documentation map

Each file answers one kind of question. Keep conclusions in these files and dated measurements in
the evidence appendix.

| File | Answers | Update when |
|---|---|---|
| [contracts.md](contracts.md) | What the safe API promises about ownership, lifetimes, unsupported operations and raw escape hatches, plus layer coverage | A public behaviour or native adapter decision changes |
| [support-matrix.md](support-matrix.md) | Which targets and options are supported, what each CI job proves, how to run ARM64 cross tests | A platform, SDK requirement or CI job changes |
| [performance.md](performance.md) | How much the wrapper costs against the native C API, method and tables | A copy path, allocation strategy or benchmark campaign changes |
| [advanced-cpu.md](advanced-cpu.md) | Runtime controls, pools, custom layers, tensor operations, readers and images | An advanced adapter gains or changes a contract |
| [model-fixtures.md](model-fixtures.md) | Which trained models are compared, with commands and numeric results | Fixtures, comparisons or their commands change |
| [evidence-history.md](evidence-history.md) | Dated campaign records: machines, counts, defects found | A new validation campaign lands |
| GitHub Releases | User-visible changes per release, generated from Conventional Commit subjects | Created when a tag is published |

## Writing rules

- State behaviour directly. Describe what the code does, not what it might do.
- Keep validation counts, machine details and log paths in the evidence appendix.
- Prefer a table over prose when the content is a matrix.
- Link the file that owns a topic instead of restating it.
