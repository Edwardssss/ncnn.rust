#![cfg(feature = "vulkan")]

//! Vulkan surface contracts.
//!
//! The cases run inside one test because the native device table is thread affine:
//! parallel tests create devices concurrently and corrupt it. A host without a
//! Vulkan device skips every case.

use ncnn_rust::gpu::Device;
use ncnn_rust::{Mat, Net};

const PARAM: &str = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n";

fn sum_net(device: &Device) -> Net {
    Net::builder()
        .expect("builder")
        .threads(1)
        .expect("threads")
        .vulkan(device.clone())
        .load_bytes(PARAM, &[])
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
