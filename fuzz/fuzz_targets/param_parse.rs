//! Feeds arbitrary bytes to every model loader: text parameters, binary parameters and the
//! bounded reader path.
//!
//! Each of them parses input that arrives from a model file, so a panic or an abort here is a
//! defect while a returned error is the expected outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ncnn_rust::{DataReader, Net};

fuzz_target!(|data: &[u8]| {
    let split = data.len() / 2;
    let text = String::from_utf8_lossy(&data[..split]);
    let binary = &data[split..];

    if let Ok(mut net) = Net::new() {
        if net.load_param(&text).is_ok() {
            let _ = net.load_model(binary);
        }
    }
    if let Ok(mut net) = Net::new() {
        if net.load_param_binary(binary).is_ok() {
            let _ = net.load_model(binary);
        }
    }
    if let (Ok(reader), Ok(mut net)) = (DataReader::from_bytes(data), Net::new()) {
        if net.load_param_reader(&reader).is_ok() {
            let _ = net.load_model_reader(&reader);
        }
    }
});