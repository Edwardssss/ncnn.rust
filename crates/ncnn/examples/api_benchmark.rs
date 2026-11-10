use ncnn::{CpuOption, Mat, Net};
use ncnn_rust as ncnn;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};
use std::{hint::black_box, time::Instant};

struct CountingAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn request(
    net: &Net,
    values: &[f32],
    mode: &str,
    input_id: &ncnn::BlobId,
    output_id: &ncnn::BlobId,
    reusable: &mut Vec<f32>,
) -> ncnn::Result<f32> {
    let input = Mat::from_f32(values, &[values.len()])?;
    let mut ex = net.extractor()?;
    if matches!(mode, "resolved" | "resolved-reused") {
        ex.input(input_id, input)?;
    } else if mode == "index" {
        ex.input(input_id.index(), input)?;
    } else if matches!(mode, "owned-view" | "owned-guard" | "guard-detach" | "name") {
        ex.input("input", input)?;
    } else {
        ex.input("input", input.deep_clone()?)?;
    }
    if mode == "resolved-reused" {
        reusable.clear();
        ex.extract_append(output_id, reusable)?;
        return Ok(black_box(reusable[0] + reusable[reusable.len() - 1]));
    }
    if mode == "owned-guard" {
        let output = ex.extract("output")?;
        let values = output.as_f32_slice()?;
        return Ok(black_box(values[0] + values[values.len() - 1]));
    }
    if mode == "guard-detach" {
        let output = ex.extract("output")?.detach()?;
        let values = output.as_f32_slice()?;
        return Ok(black_box(values[0] + values[values.len() - 1]));
    }
    let output = if mode == "index" {
        ex.extract_owned(output_id.index())?
    } else if mode == "resolved" {
        let mut values = Vec::new();
        ex.extract_into(output_id, &mut values)?;
        return Ok(black_box(values[0] + values[values.len() - 1]));
    } else {
        ex.extract_owned("output")?
    };
    if mode == "copy" {
        let values = output.values.clone();
        Ok(black_box(values[0] + values[values.len() - 1]))
    } else {
        let values = output.values;
        Ok(black_box(values[0] + values[values.len() - 1]))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("owned-view");
    if !matches!(
        mode,
        "copy"
            | "view"
            | "owned-view"
            | "owned-guard"
            | "guard-detach"
            | "name"
            | "index"
            | "resolved"
            | "resolved-reused"
            | "gpu-transfer"
            | "gpu-resident"
            | "preprocess"
    ) {
        return Err("Choose copy, view, owned-view, owned-guard, name, index, resolved, resolved-reused, guard-detach, gpu-transfer, gpu-resident, or preprocess".into());
    }
    let iterations: usize = args
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1000);
    let elements: usize = args
        .get(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(65536);
    if iterations == 0 || elements == 0 {
        return Err("Iterations and elements must be positive".into());
    }
    if mode.starts_with("gpu-") {
        #[cfg(feature = "vulkan")]
        return gpu_benchmark(mode, iterations, elements);
        #[cfg(not(feature = "vulkan"))]
        return Err("GPU modes require --features vulkan".into());
    }
    if mode == "preprocess" {
        return preprocessing(iterations, elements);
    }
    let net = Net::builder()?
        .threads(1)?
        .option(CpuOption::LightMode, 1)?
        .load_bytes(
            "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
            &[],
        )?;
    let input_id = net.resolve_blob("input")?;
    let output_id = net.resolve_blob("output")?;
    let values = vec![1f32; elements];
    let mut reusable = Vec::new();
    for _ in 0..20 {
        assert_eq!(
            request(&net, &values, mode, &input_id, &output_id, &mut reusable)?,
            2.
        );
    }
    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    let mut checksum = 0.;
    for _ in 0..iterations {
        checksum += request(&net, &values, mode, &input_id, &output_id, &mut reusable)?;
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
    let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before;
    assert_eq!(checksum, iterations as f32 * 2.);
    println!("rust_allocations={allocations}, rust_requested_bytes={allocated_bytes}, scope=measured_requests_only, excludes=native_allocations_and_copies");
    println!("mode={mode}, elements={elements}, iterations={iterations}, ns_per_request={}, checksum={checksum}", elapsed.as_nanos() / iterations as u128);
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(value) = status.lines().find(|line| line.starts_with("VmHWM:")) {
            println!("{value}");
        }
    }
    Ok(())
}

/// Times the tensor helpers that run per request, rotating the cases so a slow host warms every
/// case the same way. Each line is one median, which is what the performance page records.
#[allow(clippy::type_complexity)]
fn preprocessing(iterations: usize, elements: usize) -> Result<(), Box<dyn std::error::Error>> {
    let values: Vec<f32> = (0..elements)
        .map(|index| (index % 251) as f32 / 64.)
        .collect();
    let mean = [127.5f32; 3];
    let norm = [0.017f32; 3];
    let channel: Vec<f32> = (0..elements).map(|index| (index % 255) as f32).collect();
    let owned = Mat::from_f32(&channel, &[elements])?;
    let mut scratch = Mat::from_f32(&channel, &[elements])?;

    let side = ((elements / 3) as f64).sqrt().floor().max(2.) as usize;
    let pixels: Vec<u8> = (0..side * side * 3)
        .map(|index| (index % 251) as u8)
        .collect();
    let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let field = Mat::from_f32(&channel[..side * side * 3], &[side, side, 3])?;
    let mut image = vec![0u8; side * side * 3];

    let mut cases: Vec<(&str, Box<dyn FnMut() -> ncnn::Result<()>>)> = vec![
        (
            "borrow",
            Box::new(|| {
                let tensor = unsafe { Mat::from_external(&values, &[elements])? };
                black_box(tensor.info().w);
                Ok(())
            }),
        ),
        (
            "deep-clone",
            Box::new(|| {
                black_box(owned.deep_clone()?.info().w);
                Ok(())
            }),
        ),
        (
            "reshape-independent",
            Box::new(|| {
                black_box(owned.reshape(&[elements])?.info().w);
                Ok(())
            }),
        ),
        (
            "reshape-consuming",
            Box::new(|| {
                let tensor = unsafe { Mat::from_external(&values, &[elements])? };
                black_box(tensor.into_reshape(&[elements])?.info().w);
                Ok(())
            }),
        ),
        (
            "normalize-copy",
            Box::new(|| {
                black_box(owned.normalized(&mean[..1], &norm[..1])?.info().w);
                Ok(())
            }),
        ),
        (
            "normalize-in-place",
            Box::new(|| {
                scratch.normalize_in_place(&mean[..1], &norm[..1])?;
                Ok(())
            }),
        ),
        (
            "flatten",
            Box::new(|| {
                black_box(owned.flatten()?.info().w);
                Ok(())
            }),
        ),
        (
            "packing",
            Box::new(|| {
                black_box(owned.with_packing(4)?.info().elempack);
                Ok(())
            }),
        ),
        (
            "border",
            Box::new(|| {
                black_box(
                    field
                        .copy_make_border(1, 1, 1, 1, ncnn::BorderMode::Replicate, 0.)?
                        .info()
                        .w,
                );
                Ok(())
            }),
        ),
        (
            "warp-affine",
            Box::new(|| {
                black_box(ncnn::image::warp_affine(
                    &pixels,
                    (side, side),
                    3,
                    (side, side),
                    &identity,
                    ncnn::image::WarpBorder::Replicate,
                    0,
                )?);
                Ok(())
            }),
        ),
        (
            "to-pixels",
            Box::new(|| {
                field.to_pixels(ncnn::PixelFormat::Rgb, side * 3, &mut image)?;
                Ok(())
            }),
        ),
    ];

    const ROUNDS: usize = 5;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cases.len()];
    for (_, body) in cases.iter_mut() {
        for _ in 0..20 {
            body()?;
        }
    }
    for round in 0..ROUNDS {
        for offset in 0..cases.len() {
            let index = (offset + round) % cases.len();
            let start = Instant::now();
            for _ in 0..iterations {
                cases[index].1()?;
            }
            samples[index].push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
        }
    }
    for (index, (name, _)) in cases.iter().enumerate() {
        let mut rounds = samples[index].clone();
        rounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "preprocess={name}, elements={elements}, iterations={iterations}, rounds={ROUNDS}, median_us={:.3}",
            rounds[rounds.len() / 2]
        );
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn gpu_benchmark(
    mode: &str,
    iterations: usize,
    elements: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = ncnn::gpu::Device::default_device()?;
    let allocator = device.allocator()?;
    let net = Net::builder()?
        .vulkan(device.clone())
        .option(CpuOption::Fp16Packed, 0)?
        .option(CpuOption::Fp16Storage, 0)?
        .option(CpuOption::Fp16Arithmetic, 0)?
        .load_bytes(
            "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
            &[],
        )?;
    let input = Mat::from_f32(&vec![1.; elements], &[elements])?;
    let resident = allocator.upload(&input)?;
    let execute = || -> ncnn::Result<()> {
        if mode == "gpu-transfer" {
            let values = net
                .infer([("input", input.deep_clone()?)], &["output"])?
                .remove(0)
                .values;
            assert_eq!(black_box(values[0] + values[elements - 1]), 2.);
        } else {
            let mut command = allocator.command(&net)?;
            command.input("input", &resident)?;
            command.extract("output")?;
            let outputs = command.wait()?;
            black_box(outputs[0].storage_bytes());
        }
        Ok(())
    };
    for _ in 0..20 {
        execute()?;
    }
    let start = Instant::now();
    for _ in 0..iterations {
        execute()?;
    }
    let elapsed = start.elapsed();
    let mut command = allocator.command(&net)?;
    command.input("input", &resident)?;
    command.extract("output")?;
    assert_eq!(command.wait()?[0].download()?.to_f32()?, vec![1.; elements]);
    let (live, peak) = allocator.tensor_bytes();
    println!("device={:?}, mode={mode}, precision=fp32, warmup=20, elements={elements}, iterations={iterations}, synchronized_ns_per_request={}, tensor_live_bytes={live}, tensor_peak_bytes={peak}, device_heap_budget_mib={}", device.info(), elapsed.as_nanos() / iterations as u128, device.heap_budget_mib());
    Ok(())
}
