use ncnn::{CpuOption, Mat, Net, Options};
use ncnn_rust as ncnn;
use std::{io::Read, path::Path, process::Command};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Reference {
    name: String,
    layout: [u64; 9],
    values: Vec<f32>,
}

struct ReferenceRun {
    outputs: Vec<Reference>,
    options: [i32; 18],
    probe_index: i32,
    probe_bits: u32,
}

const OPTION_KEYS: [CpuOption; 18] = [
    CpuOption::Threads,
    CpuOption::LightMode,
    CpuOption::Winograd,
    CpuOption::Sgemm,
    CpuOption::Int8Inference,
    CpuOption::Packing,
    CpuOption::Fp16Packed,
    CpuOption::Fp16Storage,
    CpuOption::Fp16Arithmetic,
    CpuOption::Bf16Storage,
    CpuOption::LocalPool,
    CpuOption::OpenmpBlocktime,
    CpuOption::FlushDenormals,
    CpuOption::Winograd23,
    CpuOption::Winograd43,
    CpuOption::Winograd63,
    CpuOption::MappedModelLoading,
    CpuOption::KvCacheSequenceHint,
];

fn configure(options: &mut Options, threads: i32, packing: i32, profile: i32) -> ncnn::Result<()> {
    for (key, value) in [
        (CpuOption::Threads, threads),
        (CpuOption::Packing, packing),
        (CpuOption::LightMode, i32::from(profile == 4)),
        (CpuOption::Fp16Packed, i32::from(profile == 2)),
        (CpuOption::Fp16Storage, i32::from(profile == 2)),
        (CpuOption::Fp16Arithmetic, i32::from(profile == 2)),
        (CpuOption::Bf16Storage, i32::from(profile == 1)),
    ] {
        options.set(key, value)?;
    }
    if profile == 3 {
        for (key, value) in [
            (CpuOption::Winograd, 0),
            (CpuOption::Sgemm, 0),
            (CpuOption::Int8Inference, 0),
            (CpuOption::LocalPool, 0),
            (CpuOption::OpenmpBlocktime, 0),
            (CpuOption::FlushDenormals, 0),
            (CpuOption::Winograd23, 0),
            (CpuOption::Winograd43, 0),
            (CpuOption::Winograd63, 0),
            (CpuOption::MappedModelLoading, 1),
            (CpuOption::KvCacheSequenceHint, 32),
        ] {
            options.set(key, value)?;
        }
    }
    if (5..=7).contains(&profile) {
        for (key, value) in [
            (CpuOption::Sgemm, 0),
            (CpuOption::Winograd23, i32::from(profile == 5)),
            (CpuOption::Winograd43, i32::from(profile == 6)),
            (CpuOption::Winograd63, i32::from(profile == 7)),
            (CpuOption::FlushDenormals, profile - 4),
        ] {
            options.set(key, value)?;
        }
    }
    Ok(())
}

fn read<const N: usize>(input: &mut &[u8]) -> std::io::Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn reference(
    executable: &Path,
    root: &Path,
    name: &str,
    width: usize,
    height: usize,
    configuration: (i32, i32, i32),
) -> Result<ReferenceRun> {
    let (threads, packing, profile) = configuration;
    let output = Command::new(executable)
        .arg(root.join(format!("{name}.param")))
        .arg(root.join(format!("{name}.bin")))
        .args([
            width.to_string(),
            height.to_string(),
            threads.to_string(),
            packing.to_string(),
            profile.to_string(),
        ])
        .output()?;
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "Native reference failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let mut input = output.stdout.as_slice();
    assert_eq!(u32::from_le_bytes(read(&mut input)?), 0x4e435053);
    let mut options = [0; 18];
    for value in &mut options {
        *value = i32::from_le_bytes(read(&mut input)?);
    }
    let probe_index = i32::from_le_bytes(read(&mut input)?);
    let probe_bits = u32::from_le_bytes(read(&mut input)?);
    let count = u32::from_le_bytes(read(&mut input)?);
    let mut outputs = Vec::new();
    for _ in 0..count {
        let length = u32::from_le_bytes(read(&mut input)?) as usize;
        let mut name = vec![0; length];
        input.read_exact(&mut name)?;
        let mut layout = [0; 9];
        for value in &mut layout[..7] {
            *value = u32::from_le_bytes(read(&mut input)?) as u64;
        }
        for value in &mut layout[7..] {
            *value = u64::from_le_bytes(read(&mut input)?);
        }
        let count = usize::try_from(u64::from_le_bytes(read(&mut input)?))?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(f32::from_le_bytes(read(&mut input)?));
        }
        outputs.push(Reference {
            name: String::from_utf8(name)?,
            layout,
            values,
        });
    }
    assert!(input.is_empty());
    Ok(ReferenceRun {
        outputs,
        options,
        probe_index,
        probe_bits,
    })
}

fn write_int8_fixture(root: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    std::fs::write(root.join("int8.param"), "7767517\n3 3\nInput input 0 1 input\nConvolution quant 1 1 input quantized 0=4 1=1 6=12 8=101\nConvolution dequant 1 1 quantized output 0=4 1=1 6=16 8=1\n")?;
    let mut weights = Vec::new();
    for count in [12, 16] {
        weights.extend_from_slice(&0x000D4B38u32.to_le_bytes());
        weights.extend((0..count).map(|i| ((i % 5) as i8 - 2) as u8));
        for scale in [1f32, 1., 1., 1., 16.] {
            weights.extend_from_slice(&scale.to_le_bytes());
        }
        if count == 12 {
            weights.extend_from_slice(&16f32.to_le_bytes());
        }
    }
    std::fs::write(root.join("int8.bin"), weights)?;
    Ok(())
}

fn write_operator_fixture(root: &Path, name: &str, variant: usize) -> Result<()> {
    std::fs::write(root.join(format!("{name}.param")), "7767517\n6 8\nInput input 0 1 input\nReshape matrix 1 1 input matrix 0=8 1=-1\nSplit branches 1 3 matrix g l a\nGemm projection 1 1 g projected 0=0.75 1=0.5 3=1 5=1 6=1 8=8 9=8 10=4\nLayerNorm normalization 1 1 l normalized 0=8 1=0.0001 2=1\nMultiHeadAttention attention 1 1 a attended 0=8 1=2 2=64\n")?;
    let mut weights = Vec::new();
    let mut block = |count: usize, tagged: bool, seed: usize, offset: f32| {
        if tagged {
            weights.extend_from_slice(&0u32.to_le_bytes());
        }
        for i in 0..count {
            let value = (((i * 7 + seed * 3) % 23) as i32 - 11) as f32 / 32.
                * (1. + variant as f32 * 0.5)
                + offset;
            weights.extend_from_slice(&value.to_le_bytes());
        }
    };
    block(64, true, 1, 0.);
    block(8, true, 2, 0.);
    block(8, false, 3, 1.);
    block(8, false, 4, 0.);
    for seed in 5..9 {
        block(64, true, seed, 0.);
        block(8, false, seed + 4, 0.);
    }
    std::fs::write(root.join(format!("{name}.bin")), weights)?;
    Ok(())
}

fn operator_regression(root: &Path, executable: &Path) -> Result<()> {
    for (name, variant) in [("operators", 0), ("operators-reloaded", 1)] {
        write_operator_fixture(root, name, variant)?;
    }
    for (threads, packing, profile) in [(1, 0, 0), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 4)] {
        let mut net = Net::new()?;
        let mut options = Options::new()?;
        configure(&mut options, threads, packing, profile)?;
        net.configure(&options)?;
        let mut retained: Option<(ncnn::FinalOutput, Vec<f32>)> = None;
        for name in ["operators", "operators-reloaded"] {
            net.load_param_file(root.join(format!("{name}.param")))?;
            assert!(net.extractor().is_err());
            net.load_model_file(root.join(format!("{name}.bin")))?;
            assert!(net.load_model(&[]).is_err());
            assert!(net.configure(&options).is_err());
            let heights: &[usize] = if name == "operators" {
                &[2, 3, 1]
            } else {
                &[2]
            };
            for &height in heights {
                let values: Vec<_> = (0..8 * height * 3)
                    .map(|i| (((i * 17 + 3) % 251) as i32 - 125) as f32 / 64.)
                    .collect();
                let input = Mat::from_f32(&values, &[8, height, 3])?;
                let native = reference(
                    executable,
                    root,
                    name,
                    8,
                    height,
                    (threads, packing, profile),
                )?;
                assert_eq!(net.output_names(), ["projected", "normalized", "attended"]);
                assert_eq!(native.outputs.len(), 3);
                for output in &native.outputs {
                    assert_eq!(output.values.len(), values.len());
                    assert!(output.values.iter().any(|v| v.abs() > 0.01));
                    assert!(output.values.windows(2).any(|v| v[0] != v[1]));
                }
                let bits = probe_storage(&net, &input, &native)?;
                assert!(matches!(bits, 16 | 32));
                let (max_error, current) = check_runs(&net, &input, &native.outputs, true)?;
                let batch_error = check_batched(&net, &values, &[8, height, 3], &native.outputs)?;
                if height == 2 {
                    if let Some((old, values)) = &retained {
                        assert_eq!(old.values, *values);
                        assert_ne!(current.values, *values, "reloaded weights change output");
                    } else {
                        let values = current.values.clone();
                        retained = Some((current, values));
                    }
                }
                println!("{name}: shape=8x{height}x3, profile={profile}, packing={packing}, storage_bits={bits}, 3 outputs, native max absolute error={max_error}, guarded=detached, batch2 max absolute error={batch_error}");
            }
        }
        net.clear();
        assert!(net.extractor().is_err());
        net.configure(&options)?;
        drop(net);
        let (output, values) = retained.unwrap();
        assert_eq!(output.values, values);
    }
    Ok(())
}

fn probe_storage(net: &Net, input: &Mat, native: &ReferenceRun) -> Result<u32> {
    assert!(native.probe_index >= 0);
    let mut probe = net.extractor()?;
    probe.input(&net.input_names()[0], input.deep_clone()?)?;
    let mut raw = std::ptr::null_mut();
    let observed = unsafe {
        assert_eq!(
            ncnn::sys::ncnn_rs_extract(probe.as_raw(), native.probe_index, 1, &mut raw),
            0
        );
        Mat::from_raw_owned(raw, false)?
    };
    let info = observed.info();
    let bits = (info.elemsize * 8 / info.elempack as usize) as u32;
    assert_eq!(bits, native.probe_bits);
    Ok(bits)
}

fn compare(expected: &Reference, info: ncnn::TensorInfo, values: &[f32]) -> f32 {
    let layout = [
        info.dims as u64,
        info.w as u64,
        info.h as u64,
        info.d as u64,
        info.c as u64,
        info.n as u64,
        info.elempack as u64,
        info.elemsize as u64,
        info.cstep as u64,
    ];
    assert_eq!(layout, expected.layout, "{}: layout", expected.name);
    assert_eq!(values.len(), expected.values.len());
    let mut max_error = 0f32;
    for (element, (&actual, &native)) in values.iter().zip(&expected.values).enumerate() {
        let error = (actual - native).abs();
        assert!(
            actual.is_finite() && native.is_finite() && error <= 1e-5 + 1e-4 * native.abs(),
            "{}[{element}]: Rust={actual}, C++={native}",
            expected.name
        );
        max_error = max_error.max(error);
    }
    max_error
}

fn check_runs(
    net: &Net,
    input: &Mat,
    native: &[Reference],
    guarded: bool,
) -> Result<(f32, ncnn::FinalOutput)> {
    let input_name = net.input_names().remove(0);
    let mut previous = Vec::new();
    let mut detached = None;
    let mut max_error = 0f32;
    for iteration in 0..2 {
        let mut ex = net.extractor()?;
        ex.inputs([(input_name.as_str(), input.deep_clone()?)])?;
        let mut check = |index: usize, info, values: Vec<f32>| {
            max_error = max_error.max(compare(&native[index], info, &values));
            if iteration == 0 {
                previous.push(values);
            } else {
                assert_eq!(previous[index], values, "output repeatability");
            }
        };
        if iteration == 1 && guarded {
            let names: Vec<_> = native.iter().map(|output| output.name.as_str()).collect();
            let mut index = 0;
            ex.outputs_with(&names, |_, output| {
                let info = output.info();
                let values = output.to_f32()?;
                let plane = info.w as usize * info.h as usize * info.d as usize;
                for channel in 0..info.c as usize {
                    assert_eq!(
                        output.channel_f32(channel)?,
                        &values[channel * plane..(channel + 1) * plane]
                    );
                }
                if info.c > 1 && info.cstep != plane {
                    assert!(output.as_f32_slice().is_err());
                } else {
                    assert_eq!(output.as_f32_slice()?, values);
                }
                check(index, info, values);
                index += 1;
                Ok(())
            })?;
        } else {
            if !guarded {
                assert!(ex.extract(&native[0].name).is_err());
            }
            for (index, expected) in native.iter().enumerate() {
                let output = ex.extract_owned(&expected.name)?;
                check(index, output.info(), output.values.clone());
                if iteration == 0 && index == 0 {
                    detached = Some(output);
                }
            }
        }
    }
    Ok((max_error, detached.unwrap()))
}

fn check_batched(
    net: &Net,
    values: &[f32],
    dimensions: &[usize],
    native: &[Reference],
) -> Result<f32> {
    let doubled: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
    let batched = Mat::from_f32_batch(&doubled, dimensions, 2)?;
    let input_name = net.input_names().remove(0);
    let mut ex = net.extractor()?;
    ex.input(input_name.as_str(), batched)?;
    let mut max_error = 0f32;
    for expected in native {
        let output = ex.extract_owned(&expected.name)?;
        let info = output.info;
        assert_eq!(info.n, 2, "{}: batch count", expected.name);
        let layout = [
            info.dims as u64,
            info.w as u64,
            info.h as u64,
            info.d as u64,
            info.c as u64,
            1,
            info.elempack as u64,
            info.elemsize as u64,
            info.cstep as u64,
        ];
        assert_eq!(layout, expected.layout, "{}: layout", expected.name);
        let unit = expected.values.len();
        assert_eq!(
            output.values.len(),
            unit * 2,
            "{}: batch length",
            expected.name
        );
        for batch in 0..2 {
            let slice = &output.values[batch * unit..(batch + 1) * unit];
            for (element, (&actual, &native)) in slice.iter().zip(&expected.values).enumerate() {
                let error = (actual - native).abs();
                assert!(
                    error <= 1e-5 + 1e-4 * native.abs(),
                    "{} batch {batch} [{element}]: Rust={actual}, C++={native}",
                    expected.name
                );
                max_error = max_error.max(error);
            }
        }
    }
    Ok(max_error)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let root = args.next().expect(
        "Usage: model_regression <fixture-directory> <native-reference> [--compact|--vulkan]",
    );
    let executable = args
        .next()
        .expect("Provide the independently built ncnn_cpu_reference executable");
    let flag = args.next();
    let vulkan = flag.as_deref().is_some_and(|flag| flag == "--vulkan");
    let compact = match flag {
        None => false,
        Some(flag) if flag == "--compact" => true,
        Some(flag) if flag == "--vulkan" => false,
        Some(_) => return Err("Optional third argument: --compact or --vulkan".into()),
    };
    #[cfg(not(feature = "vulkan"))]
    if vulkan {
        return Err("--vulkan requires the Cargo vulkan feature".into());
    }
    #[cfg(feature = "vulkan")]
    let device = if vulkan {
        let device = ncnn::gpu::Device::default_device()?;
        eprintln!("Rust Vulkan device: {:?}", device.info());
        Some(device)
    } else {
        None
    };
    assert!(args.next().is_none(), "Unexpected extra arguments");
    let root = Path::new(&root);
    write_int8_fixture(root)?;
    for (name, width, height, expected_outputs, expected_values) in [
        ("nanodet", 320, 320, 6, 235_200),
        ("ocr", 320, 48, 1, 735_400),
        ("int8", 3, 2, 1, 24),
    ] {
        if (compact && name != "int8") || (vulkan && name == "int8") {
            continue;
        }
        let values: Vec<_> = (0..width * height * 3)
            .map(|i| (((i * 17 + 3) % 251) as i32 - 125) as f32 / 64.)
            .collect();
        let input = Mat::from_f32(&values, &[width, height, 3])?;
        let mut fp32_baseline: Option<Vec<Vec<f32>>> = None;
        for (threads, packing, profile) in [
            (1, 0, 0),
            (2, 1, 0),
            (2, 1, 1),
            (2, 1, 2),
            (2, 1, 3),
            (2, 1, 4),
            (2, 1, 5),
            (2, 1, 6),
            (2, 1, 7),
        ] {
            if (vulkan && !((threads == 1 && profile == 0) || profile == 2))
                || (profile == 4 && name == "nanodet")
                || (name == "int8" && profile >= 3 && profile != 4)
            {
                continue;
            }
            let reference = reference(
                Path::new(&executable),
                Path::new(&root),
                name,
                width,
                height,
                (
                    threads,
                    packing,
                    if vulkan { profile + 10 } else { profile },
                ),
            )?;
            let native = &reference.outputs;
            assert_eq!(native.len(), expected_outputs, "{name}: output count");
            assert_eq!(
                native
                    .iter()
                    .map(|output| output.values.len())
                    .sum::<usize>(),
                expected_values
            );
            let mut net = Net::new()?;
            #[cfg(feature = "vulkan")]
            if let Some(device) = &device {
                net.set_vulkan_device(device.clone())?;
            }
            let mut options = Options::new()?;
            configure(&mut options, threads, packing, profile)?;
            for (key, expected) in OPTION_KEYS.into_iter().zip(reference.options) {
                assert_eq!(
                    options.get(key)?,
                    expected,
                    "{key:?}: native option mapping"
                );
            }
            net.configure(&options)?;
            net.load_param_file(Path::new(&root).join(format!("{name}.param")))?;
            let model_path = Path::new(&root).join(format!("{name}.bin"));
            if profile == 3 {
                unsafe {
                    net.load_model_file_native(&model_path)?;
                }
            } else {
                net.load_model_file(&model_path)?;
            }
            let inputs = net.input_names();
            assert_eq!(inputs.len(), 1);
            assert_eq!(
                net.output_names(),
                native
                    .iter()
                    .map(|output| output.name.clone())
                    .collect::<Vec<_>>()
            );
            {
                probe_storage(&net, &input, &reference)?;
                if name == "int8" {
                    assert_eq!(reference.probe_bits, 8, "requantized convolution storage");
                } else if profile == 2 {
                    assert!(
                        matches!(reference.probe_bits, 16 | 32),
                        "{name}: native FP16 capability selection"
                    );
                } else {
                    assert_eq!(
                        reference.probe_bits,
                        if profile == 1 { 16 } else { 32 },
                        "{name}: native storage selection"
                    );
                }
            }
            let (max_error, _) = check_runs(&net, &input, native, !vulkan && profile != 3)?;
            #[cfg(feature = "vulkan")]
            if let Some(device) = &device {
                let outputs = net.infer(
                    [(&inputs[0], input.deep_clone()?)],
                    &net.output_names()
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                )?;
                let values: Vec<_> = outputs.into_iter().map(|output| output.values).collect();
                if let Some(baseline) = &fp32_baseline {
                    let mut maximum = 0f64;
                    let mut relative = 0f64;
                    let mut squares = 0f64;
                    let mut count = 0usize;
                    for (a, b) in baseline.iter().flatten().zip(values.iter().flatten()) {
                        let error = (f64::from(*a) - f64::from(*b)).abs();
                        maximum = maximum.max(error);
                        relative = relative.max(error / f64::from(a.abs()).max(1e-6));
                        squares += error * error;
                        count += 1;
                    }
                    println!("{name}: FP16-vs-FP32 max_abs={maximum}, max_relative_floor_1e-6={relative}, rmse={}", (squares / count as f64).sqrt());
                } else {
                    fp32_baseline = Some(values.clone());
                }
                let allocator = device.allocator()?;
                let resident = allocator.upload(&input)?;
                let names = net.output_names();
                let execute = || -> ncnn::Result<Vec<ncnn::gpu::Tensor>> {
                    let mut command = allocator.command(&net)?;
                    command.input(&inputs[0], &resident)?;
                    for name in &names {
                        command.extract(name)?;
                    }
                    command.wait()
                };
                for _ in 0..3 {
                    drop(execute()?);
                }
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    drop(execute()?);
                }
                let resident_us = start.elapsed().as_micros() / 10;
                let resident_outputs = execute()?;
                let mut resident_error = 0f32;
                for (output, expected) in resident_outputs.iter().zip(&values) {
                    for (a, b) in output.guard().download()?.to_f32()?.iter().zip(expected) {
                        resident_error = resident_error.max((a - b).abs());
                    }
                }
                assert!(
                    resident_error < 0.02,
                    "Resident extraction differs: {resident_error}"
                );
                for _ in 0..3 {
                    drop(net.infer(
                        [(&inputs[0], input.deep_clone()?)],
                        &names.iter().map(String::as_str).collect::<Vec<_>>(),
                    )?);
                }
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    drop(net.infer(
                        [(&inputs[0], input.deep_clone()?)],
                        &names.iter().map(String::as_str).collect::<Vec<_>>(),
                    )?);
                }
                println!("{name}: profile={profile}, warmup=3, repetitions=10, synchronized_resident_us={resident_us}, synchronized_transfer_us={}, resident_max_abs={resident_error}, tensor_bytes={:?}, device_heap_budget_mib={}", start.elapsed().as_micros()/10, allocator.tensor_bytes(), device.heap_budget_mib());
            }
            #[cfg(not(feature = "vulkan"))]
            let _ = &mut fp32_baseline;
            println!("{name}: profile={profile}, threads={threads}, packing={packing}, storage_bits={}, {expected_outputs} outputs, {expected_values} values, native max absolute error={max_error}, repeatable", reference.probe_bits);
        }
    }
    if !vulkan {
        operator_regression(root, Path::new(&executable))?;
    }
    Ok(())
}
