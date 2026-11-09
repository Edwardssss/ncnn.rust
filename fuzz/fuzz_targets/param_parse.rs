//! Feeds arbitrary bytes to the text parameter loader and the bounded weight reader.
//!
//! Both surfaces parse input that arrives from a model file, so a panic or an abort here is a
//! defect while a returned error is the expected outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ncnn_rust::Net;

fuzz_target!(|data: &[u8]| {
    let split = data.len() / 2;
    let text = String::from_utf8_lossy(&data[..split]);
    let weights = &data[split..];

    let mut net = match Net::new() {
        Ok(net) => net,
        Err(_) => return,
    };
    if net.load_param(&text).is_err() {
        return;
    }
    let _ = net.load_model(weights);
});
