# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub's
[security advisory form](https://github.com/Edwardssss/ncnn.rust/security/advisories/new). Do not open
a public issue for a vulnerability. Include the affected version, the platform, a reproducing input
and the observed failure.

The maintainer acknowledges a report within seven days and shares an assessment or a fix plan within
thirty days.

## Supported versions

The latest published minor version of `ncnn-rust` and the matching `ncnn-rust-sys` receive fixes.
Older versions are updated only when the same defect reaches them unchanged.

## Scope

This crate wraps a native inference library, so the boundary matters:

- **Model files are trusted input.** `Net::load_bytes`, `Net::load_files` and the reader APIs parse
  parameters and weights with the native parser. A malformed or hostile model can corrupt memory,
  because ncnn does not treat model bytes as untrusted. Validate models before loading them.
- **Raw handles and unsafe adapters transfer the safety proof to the caller.** `as_raw`,
  `into_raw`, `from_raw_owned`, `register_custom_layer`, the graph mutation adapters and the Vulkan
  buffer import functions are documented with their required invariants. A violated invariant is a
  caller bug, not a crate defect.
- **Memory safety of the safe API is in scope.** A safe call that reads outside its tensor, frees
  twice, leaks a handle or admits a data race is a vulnerability; report it.
- **Numerical results are not a security boundary.** Wrong values from a supported layer, a
  platform-specific rounding difference or an unsupported parameter combination are ordinary bugs,
  which belong in the public issue tracker.

## Hardening in place

- The bounded reader limits weight reads to the supplied bytes.
- Child-process isolation turns a native abort in a layer case into a named test failure.
- A nightly fuzz workflow feeds arbitrary bytes to the text parameter loader and the weight reader.
