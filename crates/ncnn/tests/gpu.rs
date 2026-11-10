#![cfg(feature = "vulkan")]

//! Vulkan surface contracts.
//!
//! The cases run inside one test because the native device table is thread affine:
//! parallel tests create devices concurrently and corrupt it. A host without a
//! Vulkan device skips every case.

use ncnn_rust::gpu::Device;
use ncnn_rust::{Endpoint, Mat, Net};

const PARAM: &str = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n";
const FLOAT_LAYERS: [(&str, &str); 1] = [(
    "Convolution",
    "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4\n",
)];
/// Quantized variants: int8 weight storage with one weight scale per output channel and one
/// input scale. The weight scale leaves `0.25` below one int8 step, so a quantized run differs
/// from the float run and both devices must agree.
const QUANTIZED_LAYERS: [(&str, &str, &[usize]); 3] = [
    (
        "Convolution",
        "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4 8=1\n",
        &[4, 4, 1],
    ),
    (
        "ConvolutionDepthWise",
        "7767517\n2 2\nInput input 0 1 input\nConvolutionDepthWise conv 1 1 input output 0=1 1=2 3=1 5=0 6=4 7=1 8=1\n",
        &[4, 4, 1],
    ),
    (
        "InnerProduct",
        "7767517\n2 2\nInput input 0 1 input\nInnerProduct ip 1 1 input output 0=1 1=0 2=4 8=1\n",
        &[4],
    ),
];
const WEIGHT_SCALE: f32 = 3.0;
/// Requantized variants keep one byte per element on the way out, so the storage of a
/// convolution itself is readable without a separate quantization layer.
const REQUANTIZED_LAYERS: [(&str, &str); 2] = [
    (
        "Convolution",
        "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4 8=101\n",
    ),
    (
        "ConvolutionDepthWise",
        "7767517\n2 2\nInput input 0 1 input\nConvolutionDepthWise conv 1 1 input output 0=1 1=2 3=1 5=0 6=4 7=1 8=101\n",
    ),
];
/// The quantized convolution followed by one-byte storage, so the stored bytes are readable.
const QUANTIZED_STORAGE: &str = "7767517\n3 3\nInput input 0 1 input\n\
     Convolution conv 1 1 input scaled 0=1 1=2 5=0 6=4 8=1\n\
     Quantize quantize 1 1 scaled stored 0=1\n";
const CONVOLUTION_INPUT: [f32; 16] = [
    -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5,
    0.75,
];

fn sum_net(device: &Device) -> Net {
    Net::builder()
        .expect("builder")
        .threads(1)
        .expect("threads")
        .vulkan(device.clone())
        .load_bytes(PARAM, &[])
        .expect("load")
}

fn tagged(values: &[f32]) -> Vec<u8> {
    let mut bytes = 0f32.to_le_bytes().to_vec();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn raw(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One float weight block, one output-channel scale and one input scale.
fn quantized_weights() -> Vec<u8> {
    [tagged(&[0.25; 4]), raw(&[WEIGHT_SCALE]), raw(&[1.0])].concat()
}

/// The same weights with the extra output scale a requantizing layer loads.
fn requantized_weights() -> Vec<u8> {
    [quantized_weights(), raw(&[1.0])].concat()
}

/// Reads the one-byte storage a requantizing layer writes.
fn stored_bytes(net: &Net, label: &str) -> Vec<i8> {
    let mut extractor = net.extractor().expect("extractor");
    extractor
        .input(Endpoint::Name("input"), conv_input())
        .expect("input");
    let guard = extractor
        .extract_stored(Endpoint::Name("output"))
        .expect("stored output");
    assert_eq!(guard.info().elemsize, 1, "{label} storage width");
    guard.to_i8().expect("stored bytes")
}

fn single_layer(body: &str, weights: &[u8], device: Option<&Device>) -> Net {
    let builder = Net::builder()
        .expect("builder")
        .threads(1)
        .expect("threads");
    match device {
        Some(device) => builder.vulkan(device.clone()),
        None => builder,
    }
    .load_bytes(body, weights)
    .expect("load")
}

fn convolution_net(device: Option<&Device>, quantized: bool) -> Net {
    if quantized {
        single_layer(QUANTIZED_LAYERS[0].1, &quantized_weights(), device)
    } else {
        single_layer(FLOAT_LAYERS[0].1, &tagged(&[0.25; 4]), device)
    }
}

fn values(mat: &Mat) -> Vec<f32> {
    mat.to_f32().expect("cpu values")
}

fn conv_input_of(shape: &[usize]) -> Mat {
    let count = shape.iter().product::<usize>();
    Mat::from_f32(&CONVOLUTION_INPUT[..count], shape).expect("convolution input")
}

fn conv_input() -> Mat {
    conv_input_of(&[4, 4, 1])
}

fn close(actual: &[f32], expected: [f32; 3], label: &str) {
    assert_eq!(actual.len(), 3, "{label} length");
    for (index, (value, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (value - want).abs() <= 1e-4,
            "{label} index {index}: {value} != {want}"
        );
    }
}

#[test]
fn vulkan_surface_contracts() {
    let devices = Device::enumerate().expect("enumerate");
    if devices.is_empty() {
        eprintln!("no Vulkan device: surface contracts are skipped");
        return;
    }

    let device = Device::default_device().expect("default device");
    let info = device.info();
    assert!(devices.iter().any(|entry| entry.index == info.index));
    assert!(!info.name.is_empty(), "device name");
    assert!(device.heap_budget_mib() > 0, "heap budget");
    assert_ne!(device.raw_device_handle(), 0, "device handle");
    assert!(Device::new(devices.len()).is_err(), "index bounds");

    let allocator = device.allocator().expect("allocator");
    assert_eq!(allocator.tensor_bytes(), (0, 0));

    let source = Mat::from_f32(&[1., 2., 3., 4.], &[4]).expect("tensor");
    let tensor = allocator.upload(&source).expect("upload");
    let (live, peak) = allocator.tensor_bytes();
    assert!(live >= 16 && peak == live, "live {live} peak {peak}");
    assert!(tensor.storage_bytes() >= 16, "storage bytes");
    assert_eq!(
        values(&tensor.download().expect("download")),
        [1., 2., 3., 4.]
    );

    let descriptor = {
        let guard = tensor.guard();
        assert_eq!(
            values(&guard.download().expect("guard download")),
            [1., 2., 3., 4.]
        );
        let descriptor = unsafe { guard.buffer_descriptor() }.expect("descriptor");
        assert_ne!(descriptor.buffer, 0, "buffer handle");
        assert!(
            descriptor.capacity >= descriptor.offset + 16,
            "buffer range"
        );
        descriptor
    };

    let imported = unsafe {
        allocator.import_buffer(
            descriptor.buffer,
            descriptor.offset,
            descriptor.capacity,
            4,
            descriptor.access,
            descriptor.stage,
        )
    };
    match imported {
        Ok(imported) => assert_eq!(
            values(&imported.download().expect("imported")),
            [1., 2., 3., 4.]
        ),
        Err(error) => eprintln!("external buffer import rejected by the driver: {error:?}"),
    }

    drop(tensor);
    assert_eq!(allocator.tensor_bytes().0, 0, "live bytes after drop");

    let net = sum_net(&device);
    let a = allocator
        .upload(&Mat::from_f32(&[1., 2., 3.], &[3]).expect("a"))
        .expect("upload a");
    let b = allocator
        .upload(&Mat::from_f32(&[4., 5., 6.], &[3]).expect("b"))
        .expect("upload b");
    let mut command = allocator.command(&net).expect("command");
    command.input("a", &a).expect("input a");
    command.input("b", &b).expect("input b");
    let index = command.extract("sum").expect("extract");
    let outputs = command.wait().expect("wait");
    close(
        &values(&outputs[index].download().expect("output")),
        [5., 7., 9.],
        "resident sum",
    );
    assert!(
        allocator.tensor_bytes().1 >= 36,
        "peak across the recording"
    );
    drop(outputs);

    // Quantized layers read a float weight block followed by one scale per output channel and
    // one input scale. The weight scale leaves `0.25` below one int8 step, so a quantized run
    // has to differ from the float run and still agree between host and device.
    let float_run = convolution_net(None, false)
        .infer([("input", conv_input())], &["output"])
        .expect("float convolution")
        .remove(0)
        .values;
    for (label, body, shape) in QUANTIZED_LAYERS {
        let weights = quantized_weights();
        let host_run = single_layer(body, &weights, None)
            .infer([("input", conv_input_of(shape))], &["output"])
            .expect("quantized host run")
            .remove(0)
            .values;
        let device_run = single_layer(body, &weights, Some(&device))
            .infer([("input", conv_input_of(shape))], &["output"])
            .expect("quantized device run")
            .remove(0)
            .values;
        assert!(
            (host_run[0] - float_run[0]).abs() > 0.5,
            "{label}: quantized weights must change the result: {} vs {}",
            host_run[0],
            float_run[0]
        );
        for (index, (device_value, host_value)) in device_run.iter().zip(&host_run).enumerate() {
            assert!(
                (device_value - host_value).abs() <= 1e-2,
                "{label} index {index}: {device_value} != {host_value}"
            );
        }
    }

    // The quantized graph keeps one byte per element after `Quantize`, and a device network
    // downloads that copy before the guarded borrow, so the stored bytes are readable there.
    let storage_weights = [quantized_weights(), raw(&[1.0])].concat();
    let device_storage = single_layer(QUANTIZED_STORAGE, &storage_weights, Some(&device));
    let stored = {
        let mut extractor = device_storage.extractor().expect("device extractor");
        extractor
            .input(Endpoint::Name("input"), conv_input())
            .expect("device input");
        let guard = extractor
            .extract_stored(Endpoint::Name("stored"))
            .expect("device stored output");
        assert_eq!(guard.info().elemsize, 1, "device storage width");
        guard.to_i8().expect("device stored bytes")
    };
    let host_storage = single_layer(QUANTIZED_STORAGE, &storage_weights, None);
    let host_stored = {
        let mut extractor = host_storage.extractor().expect("host extractor");
        extractor
            .input(Endpoint::Name("input"), conv_input())
            .expect("host input");
        let guard = extractor
            .extract_stored(Endpoint::Name("stored"))
            .expect("host stored output");
        guard.to_i8().expect("host stored bytes")
    };
    assert_eq!(stored, host_stored, "device and host stored bytes");

    // A requantizing layer writes one byte per element itself, so the convolution storage is
    // readable without the extra quantization layer.
    for (label, body) in REQUANTIZED_LAYERS {
        let weights = requantized_weights();
        let host = stored_bytes(&single_layer(body, &weights, None), label);
        assert!(
            host.iter().any(|value| *value != 0),
            "{label}: requantized output is not all zeros"
        );
        let device_values = stored_bytes(&single_layer(body, &weights, Some(&device)), label);
        assert_eq!(host, device_values, "{label}: device and host bytes");
    }

    let quantized_net = convolution_net(Some(&device), true);

    let uploaded = allocator
        .upload(&conv_input())
        .expect("upload convolution input");
    let mut quantized_command = allocator
        .command(&quantized_net)
        .expect("quantized command");
    quantized_command
        .input("input", &uploaded)
        .expect("quantized input");
    let quantized_index = quantized_command
        .extract("output")
        .expect("quantized extract");
    let quantized_outputs = quantized_command.wait().expect("quantized wait");
    let resident_run = values(
        &quantized_outputs[quantized_index]
            .download()
            .expect("quantized output"),
    );
    let host_conv = single_layer(QUANTIZED_LAYERS[0].1, &quantized_weights(), None)
        .infer([("input", conv_input())], &["output"])
        .expect("quantized host run")
        .remove(0)
        .values;
    for (index, (resident_value, host_value)) in resident_run.iter().zip(&host_conv).enumerate() {
        assert!(
            (resident_value - host_value).abs() <= 1e-2,
            "resident quantized convolution index {index}: {resident_value} != {host_value}"
        );
    }
    drop(quantized_outputs);

    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = directory.join("gpu-pipeline-cache.bin");
    device.save_pipeline_cache(&path).expect("save cache");
    assert!(
        std::fs::metadata(&path).expect("metadata").len() > 0,
        "cache payload"
    );
    device.load_pipeline_cache(&path).expect("load cache");
    let bytes = device.pipeline_cache().expect("serialize cache");
    assert_eq!(
        bytes,
        std::fs::read(&path).expect("read cache"),
        "memory form"
    );
    device
        .set_pipeline_cache(&bytes)
        .expect("load memory cache");
    std::fs::write(&path, b"not a cache").expect("write");
    assert!(device.load_pipeline_cache(&path).is_err(), "damaged cache");
    assert!(
        device.set_pipeline_cache(b"not a cache").is_err(),
        "damaged memory cache"
    );
    std::fs::remove_file(&path).expect("remove cache");

    if devices.len() < 2 {
        eprintln!("single Vulkan device: the cross-device check is skipped");
        return;
    }
    let first = Device::new(0).expect("first device");
    let second = Device::new(1).expect("second device");
    let other_net = sum_net(&second);
    assert!(
        first
            .allocator()
            .expect("allocator")
            .command(&other_net)
            .is_err(),
        "allocator and network must share a device"
    );
}
