//! Reads quantized storage and builds a typed tensor from raw bytes.
//!
//! `extract_stored` keeps the element width a layer wrote, so the bytes travel to Rust as they
//! are; `from_typed` goes the other way round. Native `Dequantize` expects 32-bit integers, so a
//! one-byte tensor is read through `to_i8` rather than through that layer.

use ncnn_rust::{ElementKind, Mat, Net};

const GRAPH: &str = "7767517\n2 2\nInput input 0 1 input\nQuantize quantize 1 1 input stored 0=1\n";

fn main() -> ncnn_rust::Result<()> {
    // Quantize reads one raw scale block.
    let weights = 1.0f32.to_le_bytes();
    let net = Net::builder()?.threads(1)?.load_bytes(GRAPH, &weights)?;

    {
        let mut extractor = net.extractor()?;
        extractor.input("input", Mat::from_f32(&[0.1, 0.4, -0.2, 0.6], &[2, 2])?)?;
        let guard = extractor.extract_stored("stored")?;
        println!("stored width: {} bytes", guard.info().elemsize);
        println!("stored bytes: {:?}", guard.to_i8()?);
    }

    // The whole-request form keeps the same storage and reports each output's metadata.
    let outputs = net.infer_stored(
        [("input", Mat::from_f32(&[0.1, 0.4, -0.2, 0.6], &[2, 2])?)],
        &["stored"],
    )?;
    let (info, stored) = &outputs[0];
    println!(
        "infer_stored: {} bytes -> {:?}",
        info.elemsize,
        stored.to_i8()?
    );

    // The same bytes a quantized model expects on its input side.
    let typed = Mat::from_typed(ElementKind::I8, &[1, 2, 3, 0xFF], &[2, 2])?;
    println!("typed input: {:?}", typed.to_i8()?);
    println!("typed as fp32: {:?}", typed.to_f32_typed(ElementKind::I8)?);

    // A reader that names the wrong format explains the mismatch instead of failing silently.
    match typed.to_f32_typed(ElementKind::F32) {
        Ok(values) => println!("unexpected f32 read: {values:?}"),
        Err(error) => println!("wrong format: {error}"),
    }
    Ok(())
}
