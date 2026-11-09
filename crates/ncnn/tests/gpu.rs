#![cfg(feature = "vulkan")]

//! Vulkan surface contracts.
//!
//! The cases run inside one test because the native device table is thread affine:
//! parallel tests create devices concurrently and corrupt it. A host without a
//! Vulkan device skips every case.

use ncnn_rust::gpu::Device;
use ncnn_rust::{Mat, Net};

const PARAM: &str = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n";
const FLOAT_CONVOLUTION: &str =
    "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4\n";
const QUANTIZED_CONVOLUTION: &str =
    "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4 8=1\n";
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
    let mut bytes = vec![0f32.to_le_bytes()].concat();
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

/// A one-channel two by two convolution, with int8 weight storage when `quantized`.
fn convolution_net(device: Option<&Device>, quantized: bool) -> Net {
    let (body, weights) = if quantized {
        (
            QUANTIZED_CONVOLUTION,
            [tagged(&[0.25; 4]), raw(&[3.0]), raw(&[1.0])].concat(),
        )
    } else {
        (FLOAT_CONVOLUTION, tagged(&[0.25; 4]))
    };
    let builder = Net::builder()
        .expect("builder")
        .threads(1)
        .expect("threads");
    match device {
        Some(device) => builder.vulkan(device.clone()),
        None => builder,
    }
    .load_bytes(body, &weights)
    .expect("load")
}

fn values(mat: &Mat) -> Vec<f32> {
    mat.to_f32().expect("cpu values")
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
    drop(guard);

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

    // A quantized convolution reads a float weight block followed by one scale per output
    // channel and one input scale. The weight scale leaves `0.25` below one int8 step, so a
    // quantized run has to differ from the float run and still agree between host and device.
    let input = || Mat::from_f32(&CONVOLUTION_INPUT, &[4, 4, 1]).expect("convolution input");
    let float_run = convolution_net(None, false)
        .infer([("input", input())], &["output"])
        .expect("float convolution")
        .remove(0)
        .values;
    let host_run = convolution_net(None, true)
        .infer([("input", input())], &["output"])
        .expect("quantized convolution")
        .remove(0)
        .values;
    let quantized_net = convolution_net(Some(&device), true);
    let device_run = quantized_net
        .infer([("input", input())], &["output"])
        .expect("device quantized convolution")
        .remove(0)
        .values;
    assert!(
        (host_run[0] - float_run[0]).abs() > 0.5,
        "quantized weights must change the result: {} vs {}",
        host_run[0],
        float_run[0]
    );
    for (index, (device_value, host_value)) in device_run.iter().zip(&host_run).enumerate() {
        assert!(
            (device_value - host_value).abs() <= 1e-2,
            "quantized convolution index {index}: {device_value} != {host_value}"
        );
    }

    let uploaded = allocator
        .upload(&input())
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
    for (index, (resident_value, host_value)) in resident_run.iter().zip(&host_run).enumerate() {
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
    std::fs::write(&path, b"not a cache").expect("write");
    assert!(device.load_pipeline_cache(&path).is_err(), "damaged cache");
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
