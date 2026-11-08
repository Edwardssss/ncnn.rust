use ncnn_rust::{Mat, Net};

fn main() -> ncnn_rust::Result<()> {
    let builder = Net::builder()?.threads(1)?;
    let use_vulkan = std::env::args().any(|arg| arg == "--vulkan");
    #[cfg(feature = "vulkan")]
    let builder = if use_vulkan {
        let device = ncnn_rust::gpu::Device::default_device()?;
        println!(
            "Vulkan backend with per-layer CPU fallback: {:?}",
            device.info()
        );
        builder.vulkan(device)
    } else {
        builder
    };
    #[cfg(not(feature = "vulkan"))]
    if use_vulkan {
        return Err(ncnn_rust::Error::Unsupported(
            "Build with --features vulkan",
        ));
    }
    let net = builder.load_bytes(
        "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n",
        &[],
    )?;
    let input_a = net.resolve_blob("a")?;
    let input_b = net.resolve_blob("b")?;
    let sum = net.resolve_blob("sum")?;
    let mut values = Vec::with_capacity(3);
    for _ in 0..2 {
        let mut extractor = net.extractor()?;
        extractor.input(&input_a, Mat::from_f32(&[1., 2., 3.], &[3])?)?;
        extractor.input(&input_b, Mat::from_f32(&[4., 5., 6.], &[3])?)?;
        values.clear();
        extractor.extract_append(&sum, &mut values)?;
        println!("{values:?}");
    }
    if !use_vulkan {
        let mut extractor = net.extractor()?;
        extractor.inputs([
            ("a", Mat::from_f32(&[1., 2., 3.], &[3])?),
            ("b", Mat::from_f32(&[4., 5., 6.], &[3])?),
        ])?;
        extractor.outputs_with(&["sum"], |name, output| {
            println!("{name}: {:?}", output.as_f32_slice()?);
            Ok(())
        })?;
        let detached = extractor.extract_owned("sum")?;
        drop(extractor);
        drop(net);
        println!("Independent result: {:?}", detached.values);
    }
    Ok(())
}
