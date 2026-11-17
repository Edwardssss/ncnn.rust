//! Feeds arbitrary bytes to every model loader: text parameters, binary parameters and the
//! bounded reader path.
//!
//! Each of them parses input that arrives from a model file, so a panic or an abort here is a
//! defect while a returned error is the expected outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ncnn_rust::{DataReader, Net, ParamSource, WeightSource};

fuzz_target!(|data: &[u8]| {
    let split = data.len() / 2;
    let text = String::from_utf8_lossy(&data[..split]);
    let binary = &data[split..];

    if let Ok(mut net) = Net::new() {
        if net.load_param(ParamSource::Text(text.as_ref())).is_ok() {
            let _ = net.load_model(WeightSource::Bytes(binary));
        }
    }
    if let Ok(mut net) = Net::new() {
        if net.load_param(ParamSource::Binary(binary)).is_ok() {
            let _ = net.load_model(WeightSource::Bytes(binary));
        }
    }
    if let (Ok(reader), Ok(mut net)) = (DataReader::from_bytes(data), Net::new()) {
        if net.load_param(ParamSource::Reader(&reader)).is_ok() {
            let _ = net.load_model(WeightSource::Reader(&reader));
        }
    }
});
