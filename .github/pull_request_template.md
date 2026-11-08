## Summary

<!-- What changes, and why. -->

## Public API impact

<!-- New, changed or removed items. Write "none" when the surface is unchanged. -->

## Validation

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --features vulkan -- -D warnings`
- [ ] `cargo test --workspace --features vulkan`
- [ ] CPU-only `cargo test --workspace`
- [ ] `cargo run --release -p ncnn-rust --example api_benchmark` when copy paths or timings changed
- [ ] Documentation updated for changed contracts

## Platform evidence

<!-- Operating system, ncnn version, backend and hardware actually exercised. -->
