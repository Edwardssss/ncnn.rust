//! Uploads inputs, records one inference and reads the resident result back.
//!
//! The resident path keeps tensors on the device between work submissions. Building without the
//! `vulkan` feature leaves only the message below.

#[cfg(feature = "vulkan")]
fn main() -> ncnn_rust::Result<()> {
    use ncnn_rust::gpu::Device;
    use ncnn_rust::{Mat, Net};

    let device = Device::default_device()?;
    println!("device: {:?}", device.info());

    let net = Net::builder()?
        .threads(1)?
        .vulkan(device.clone())
        .load_bytes(
            "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n",
            &[],
        )?;

    let allocator = device.allocator()?;
    let a = allocator.upload(&Mat::from_f32(&[1., 2., 3.], &[3])?)?;
    let b = allocator.upload(&Mat::from_f32(&[4., 5., 6.], &[3])?)?;

    let mut command = allocator.command(&net)?;
    command.input("a", &a)?;
    command.input("b", &b)?;
    let sum = command.extract("sum")?;
    let outputs = command.wait()?;
    println!("resident sum: {:?}", outputs[sum].download()?.to_f32()?);
    drop(outputs);
    println!("allocator bytes: {:?}", allocator.tensor_bytes());
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("Build with --features vulkan to run this example");
}
