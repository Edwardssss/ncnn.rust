use ncnn_rust as ncnn;
mod advanced;

use ncnn::{
    sys, CpuOption, Layer, Mat, ModelBin, Net, Options, ParamDict, PixelFormat, PoolAllocator,
};

const GRAPH: &str = "7767517\n4 4\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\nBinaryOp product 2 1 a b product 0=2\n";

fn native_code<T>(result: ncnn::Result<T>) -> Option<i32> {
    result.err().and_then(|error| error.native_code())
}

#[test]
fn independent_native_preferences() -> ncnn::Result<()> {
    use ncnn::OptionFlag::*;
    let legacy = Options::new()?;
    let mut native = Options::native_defaults()?;
    assert!(!legacy.flag(LightMode)?);
    assert!(native.flag(LightMode)?);
    for flag in [
        SubgroupOps,
        Int8Packed,
        Int8Storage,
        Int8Arithmetic,
        Bf16Packed,
        TensorStorage,
        WeightsInHostMemory,
        ShaderLocalMemory,
        CooperativeMatrix,
        A53A55OptimizedKernel,
        Fp16Uniform,
        Int8Uniform,
        Int16Packed,
        Int16Storage,
    ] {
        native.set_flag(flag, false)?;
        assert!(!native.flag(flag)?);
        native.set_flag(flag, true)?;
        assert!(native.flag(flag)?);
    }
    native.set_flag(Fp16Storage, false)?;
    assert!(native.flag(Fp16Arithmetic)?);
    assert!(!native.flag(Fp16Storage)?);
    let net = ncnn::NetBuilder::with_options(Options::native_defaults()?)
        .threads(1)?
        .flag(LightMode, false)?
        .load_bytes(GRAPH, &[])?;
    let detached = net.infer(
        [
            ("a", Mat::from_f32(&[1., 2.], &[2])?),
            ("b", Mat::from_f32(&[3., 4.], &[2])?),
        ],
        &["sum"],
    )?;
    drop(net);
    assert_eq!(detached[0].values, [4., 6.]);
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_contracts() -> ncnn::Result<()> {
    use ncnn::gpu::{Device, Precision};
    let devices = Device::enumerate()?;
    if devices.is_empty() {
        assert!(Device::default_device().is_err());
        eprintln!("No Vulkan device: runtime inference coverage is unavailable");
        return Ok(());
    }
    let device = Device::default_device()?;
    eprintln!("Vulkan device: {:?}", device.info());
    assert!(Device::new(usize::MAX).is_err());
    assert!(Device::new(devices.len()).is_err());
    let a = Mat::from_f32(&[1., 2., 3., 4.], &[4])?;
    let b = Mat::from_f32(&[2., 3., 4., 5.], &[4])?;
    let mut reference = [0.; 4];
    unsafe {
        assert_eq!(
            sys::ncnn_rs_reference_vulkan_add(
                device.info().index as i32,
                a.to_f32()?.as_ptr(),
                b.to_f32()?.as_ptr(),
                4,
                reference.as_mut_ptr()
            ),
            0
        );
    }
    assert_eq!(reference, [3., 5., 7., 9.]);
    let mut options = Options::new()?;
    options.vulkan_precision(Precision::Fp32)?;
    options.set_flag(ncnn::OptionFlag::Fp16Storage, true)?;
    assert!(options.flag(ncnn::OptionFlag::Fp16Storage)?);
    assert!(!options.flag(ncnn::OptionFlag::Fp16Arithmetic)?);
    options.vulkan_precision(Precision::Fp16)?;
    assert!(options.flag(ncnn::OptionFlag::Fp16Arithmetic)?);
    assert!(!options.flag(ncnn::OptionFlag::Bf16Storage)?);
    options.vulkan_precision(Precision::Fp32)?;
    let mut survivor = Net::builder()?
        .vulkan(device.clone())
        .load_bytes(GRAPH, &[])?;
    for precision in [Precision::Fp32, Precision::Fp16] {
        options.vulkan_precision(precision)?;
        let detached = {
            let mut net = Net::new()?;
            assert!(net.vulkan_device().is_none());
            net.set_vulkan_device(device.clone())?;
            net.configure(&options)?;
            assert!(net.load_param_binary(&[]).is_err());
            assert!(net.vulkan_device().is_some());
            net.load_param(
                "7767517\n2 2\nInput input 0 1 input\nInnerProduct fc 1 1 input output 0=1 2=3\n",
            )?;
            assert!(net.load_model(&[]).is_err());
            assert!(net.extractor().is_err());
            assert!(net.vulkan_device().is_some());
            net.load_param(GRAPH)?;
            net.load_model(&[])?;
            assert!(net.set_vulkan_device(device.clone()).is_err());
            let mut ex = net.extractor()?;
            ex.input("a", a.deep_clone()?)?;
            ex.input("b", b.deep_clone()?)?;
            assert!(ex.extract("sum").is_err());
            let mut copied = Vec::with_capacity(16);
            ex.extract_into("sum", &mut copied)?;
            ex.extract_append("product", &mut copied)?;
            assert_eq!(copied, [3., 5., 7., 9., 2., 6., 12., 20.]);
            assert!(ex.extract_into(-1, &mut copied).is_err());
            let sum = ex.extract_owned("sum")?;
            assert_eq!(ex.extract_owned("product")?.values, [2., 6., 12., 20.]);
            sum
        };
        assert_eq!(detached.values, reference);
        assert_eq!(
            survivor.infer([("a", a.deep_clone()?), ("b", b.deep_clone()?)], &["sum"])?[0].values,
            reference
        );
    }
    let downloaded = {
        let temporary = device.allocator()?;
        let tensor = temporary.upload(&a)?;
        let first = tensor.download()?;
        let second = tensor.download()?;
        unsafe {
            *(sys::ncnn_mat_get_data(first.as_raw()) as *mut f32) = -99.;
        }
        assert_eq!(second.to_f32()?, a.to_f32()?);
        second
    };
    assert_eq!(downloaded.to_f32()?, a.to_f32()?);
    let allocator = device.allocator()?;
    let resident_a = allocator.upload(&a)?;
    let resident_b = allocator.upload(&b)?;
    assert!(resident_a.storage_bytes() >= 16);
    let padded = Mat::from_f32(&[1., 2., 3., 4., 5., 6.], &[1, 1, 2, 3])?;
    assert_eq!(
        allocator.upload(&padded)?.download()?.to_f32()?,
        padded.to_f32()?
    );
    let guard = resident_a.guard();
    assert_eq!(guard.download()?.to_f32()?, a.to_f32()?);
    let descriptor = unsafe { guard.buffer_descriptor()? };
    let imported = unsafe {
        allocator.import_buffer(
            descriptor.buffer,
            descriptor.offset,
            descriptor.capacity,
            4,
            descriptor.access,
            descriptor.stage,
        )?
    };
    assert_eq!(imported.download()?.to_f32()?, a.to_f32()?);
    assert!(unsafe {
        allocator.import_buffer(
            descriptor.buffer,
            descriptor.offset + 1,
            descriptor.capacity,
            4,
            descriptor.access,
            descriptor.stage,
        )
    }
    .is_err());
    drop(imported);
    assert!(allocator.tensor_bytes().0 >= 32);
    assert_eq!(resident_a.download()?.to_f32()?, a.to_f32()?);
    let mut command = allocator.command(&survivor)?;
    command.input("a", &resident_a)?;
    command.input("b", &resident_b)?;
    command.extract("sum")?;
    command.extract("product")?;
    assert!(command.input("a", &resident_a).is_err());
    let outputs = command.wait()?;
    assert_eq!(outputs[0].download()?.to_f32()?, reference);
    assert_eq!(outputs[1].download()?.to_f32()?, [2., 6., 12., 20.]);
    drop(allocator.command(&survivor)?);
    let mut cancelled = allocator.command(&survivor)?;
    cancelled.input("a", &resident_a)?;
    cancelled.input("b", &resident_b)?;
    cancelled.extract("sum")?;
    drop(cancelled);
    let mut failed_input = allocator.command(&survivor)?;
    assert!(failed_input.input("missing", &resident_a).is_err());
    assert!(failed_input.extract("sum").is_err());
    assert!(failed_input.wait().is_err());
    let mut failed_output = allocator.command(&survivor)?;
    failed_output.input("a", &resident_a)?;
    failed_output.input("b", &resident_b)?;
    assert!(failed_output.extract("missing").is_err());
    assert!(failed_output.wait().is_err());
    drop(allocator);
    survivor.clear();
    assert_eq!(outputs[0].download()?.to_f32()?, reference);
    // The cache file lives in the system temp directory so the test also runs from
    // an extracted crate archive, which has no workspace target directory.
    let cache = std::env::temp_dir().join("ncnn-rust-resident-vulkan.cache");
    device.save_pipeline_cache(&cache)?;
    device.load_pipeline_cache(&cache)?;
    let valid = std::fs::read(&cache)?;
    for length in [0, 1, valid.len() / 2, valid.len() - 1] {
        std::fs::write(&cache, &valid[..length])?;
        assert!(device.load_pipeline_cache(&cache).is_err());
    }
    let mut payload = valid.clone();
    *payload.last_mut().unwrap() ^= 1;
    std::fs::write(&cache, payload)?;
    assert!(device.load_pipeline_cache(&cache).is_err());
    let mut bytes = valid;
    bytes[0] ^= 1;
    std::fs::write(&cache, bytes)?;
    assert!(device.load_pipeline_cache(&cache).is_err());
    std::fs::remove_file(&cache)?;
    survivor.load_param(GRAPH)?;
    survivor.load_model(&[])?;
    assert_eq!(
        survivor.infer([("a", a.deep_clone()?), ("b", b.deep_clone()?)], &["sum"])?[0].values,
        reference
    );
    Ok(())
}

#[test]
fn cpu_contracts() -> ncnn::Result<()> {
    assert!(ncnn::version().contains("20261020"), "{}", ncnn::version());
    assert!(ncnn::cpu_count() > 0);
    let a = Mat::from_f32(&[1., 2., 3., 4.], &[4])?;
    let b = Mat::from_f32(&[2., 3., 4., 5.], &[4])?;
    assert!(Mat::from_f32(&[1.], &[usize::MAX]).is_err());
    assert!(Mat::from_f32(&[1.], &[2]).is_err());
    let padded = Mat::from_f32(&[1., 2., 3., 4., 5., 6.], &[1, 1, 2, 3])?;
    assert_eq!(padded.to_f32()?, vec![1., 2., 3., 4., 5., 6.]);
    assert_eq!(padded.reshape(&[6])?.to_f32()?, padded.to_f32()?);
    assert_eq!(a.normalized(&[1.], &[2.])?.to_f32()?, vec![0., 2., 4., 6.]);
    assert_eq!(a.to_f32()?, vec![1., 2., 3., 4.]);
    for dims in [&[6][..], &[2, 3][..], &[1, 2, 3][..], &[1, 1, 2, 3][..]] {
        let original = Mat::from_f32(&[1., 2., 3., 4., 5., 6.], dims)?;
        let channels = original.info().c as usize;
        let mean: Vec<_> = (0..channels).map(|q| q as f32 + 1.).collect();
        let normalized = original.normalized(&mean, &vec![2.; channels])?;
        let expected: Vec<_> = (0..6)
            .map(|i| (i as f32 + 1. - mean[i / (6 / channels)]) * 2.)
            .collect();
        assert_eq!(normalized.to_f32()?, expected);
        assert_eq!(normalized.info().dims, original.info().dims);
        assert_eq!(normalized.info().c, original.info().c);
        assert_eq!(original.to_f32()?, [1., 2., 3., 4., 5., 6.]);
        assert_eq!(original.normalized(&[], &[])?.to_f32()?, original.to_f32()?);
        assert_eq!(
            original.normalized(&[], &vec![2.; channels])?.to_f32()?,
            [2., 4., 6., 8., 10., 12.]
        );
        assert_eq!(original.reshape_view(&[6])?.to_f32()?, original.to_f32()?);
        let independent = original.reshape(&[2, 3])?;
        unsafe {
            *(sys::ncnn_mat_get_data(independent.as_raw()) as *mut f32) = 99.;
        }
        assert_eq!(original.to_f32()?[0], 1.);
    }
    let mut reusable = Vec::with_capacity(32);
    let allocation = reusable.as_ptr();
    padded.to_f32_into(&mut reusable)?;
    assert_eq!(reusable, [1., 2., 3., 4., 5., 6.]);
    a.to_f32_into(&mut reusable)?;
    assert_eq!(reusable.as_ptr(), allocation);
    padded.append_f32_to(&mut reusable)?;
    assert_eq!(reusable, [1., 2., 3., 4., 1., 2., 3., 4., 5., 6.]);
    let mut wrong = [42.; 3];
    assert!(a.copy_to_f32(&mut wrong).is_err());
    assert_eq!(wrong, [42.; 3]);
    assert_eq!(
        Mat::from_pixels(&[10, 20, 30], PixelFormat::Rgb, 1, 1)?.to_f32()?,
        vec![10., 20., 30.]
    );
    let mut reference = [0.; 4];
    unsafe {
        assert_eq!(
            sys::ncnn_rs_reference_add(
                a.to_f32()?.as_ptr(),
                b.to_f32()?.as_ptr(),
                4,
                reference.as_mut_ptr()
            ),
            0
        );
    }
    let net = Net::builder()?.threads(1)?.load_bytes(GRAPH, &[])?;
    let mut outputs = Vec::with_capacity(32);
    let output_allocation = outputs.as_ptr();
    net.infer_into(
        [("a", a.deep_clone()?), ("b", b.deep_clone()?)],
        &["sum", "product"],
        &mut outputs,
    )?;
    assert_eq!(outputs, [3., 5., 7., 9., 2., 6., 12., 20.]);
    assert_eq!(outputs.as_ptr(), output_allocation);
    assert!(net
        .infer_into([("missing", a.deep_clone()?)], &["sum"], &mut outputs)
        .is_err());
    assert_eq!(outputs, [3., 5., 7., 9., 2., 6., 12., 20.]);
    assert!(net
        .infer_into(
            [("a", a.deep_clone()?), ("b", b.deep_clone()?)],
            &["sum", "missing"],
            &mut outputs
        )
        .is_err());
    assert_eq!(outputs, [3., 5., 7., 9.]);
    {
        let shared = a.deep_clone()?;
        let raw = shared.as_raw();
        let borrowed = unsafe {
            Mat::from_raw_owned(sys::ncnn_mat_reshape_1d(raw, 4, std::ptr::null_mut()), true)?
        };
        let mut ex = net.extractor()?;
        ex.input("a", shared)?;
        assert_eq!(
            borrowed.normalized(&[1.], &[2.])?.to_f32()?,
            [0., 2., 4., 6.]
        );
        assert_eq!(borrowed.reshape_view(&[2, 2])?.to_f32()?, [1., 2., 3., 4.]);
        ex.input("b", b.deep_clone()?)?;
        assert_eq!(ex.extract_owned("sum")?.values, [3., 5., 7., 9.]);
    }
    assert_eq!(
        net.infer([("a", a.deep_clone()?), ("b", b.deep_clone()?)], &["sum"])?[0].values,
        [3., 5., 7., 9.]
    );
    {
        let mut ex = net.extractor()?;
        ex.input("a", a.deep_clone()?)?;
        ex.input("b", b.deep_clone()?)?;
        let exact = [3., 5., 7., 9.];
        let mut copy = Vec::new();
        ex.extract_into("sum", &mut copy)?;
        assert_eq!(copy, exact);
        assert!(ex.extract_into("missing", &mut outputs).is_err());
        assert_eq!(outputs, [3., 5., 7., 9.]);
        ex.extract_into("product", &mut outputs)?;
    }
    drop(net);
    assert_eq!(outputs, [2., 6., 12., 20.]);
    for _ in 0..8 {
        let detached = {
            let mut net = Net::new()?;
            let mut opt = Options::new()?;
            assert!(opt.set(CpuOption::Threads, 0).is_err());
            opt.set(CpuOption::Threads, 1)?;
            opt.set(CpuOption::LightMode, 0)?;
            assert_eq!(opt.get(CpuOption::Threads)?, 1);
            net.configure(&opt)?;
            net.load_param(GRAPH)?;
            net.load_model(&[])?;
            assert_eq!(net.input_names(), ["a", "b"]);
            assert_eq!(net.output_names(), ["sum", "product"]);
            assert!(net.configure(&opt).is_err());
            let mut ex = net.extractor()?;
            ex.input("a", a.deep_clone()?)?;
            ex.input(1, b.deep_clone()?)?;
            assert!(ex.extract_owned("missing").is_err());
            assert!(ex.extract_owned(i32::MAX).is_err());
            let sum = ex.extract_owned("sum")?;
            assert_eq!(sum.values, reference);
            assert_eq!(ex.extract_owned(3)?.values, vec![2., 6., 12., 20.]);
            sum
        };
        assert_eq!(detached.values, reference);
    }
    let mut net = Net::new()?;
    assert!(net.load_param_binary(&[]).is_err());
    assert!(net.extractor().is_err());
    let pool = PoolAllocator::new()?;
    assert!(!pool.as_raw().is_null());
    let mut pd = ParamDict::new()?;
    pd.set_float(0, 0.)?;
    assert_eq!(pd.get_float(0, 1.)?, 0.);
    assert!(pd.set_int(32, 0).is_err());
    let weights = ModelBin::from_mats(&[])?;
    let mut relu = Layer::new("ReLU", Options::new()?)?;
    unsafe {
        relu.load_param(&pd)?;
        relu.load_model(weights.as_raw())?;
        relu.create_pipeline()?;
        let input = Mat::from_f32(&[-2., 0., 3.], &[3])?;
        assert_eq!(relu.forward(&input)?.to_f32()?, vec![0., 0., 3.]);
    }
    let mut fp32 = Options::new()?;
    fp32.set(CpuOption::Fp16Storage, 0)?;
    fp32.set(CpuOption::Fp16Packed, 0)?;
    fp32.set(CpuOption::Fp16Arithmetic, 0)?;
    let mut weighted = Layer::new("InnerProduct", fp32)?;
    let mut ip = ParamDict::new()?;
    ip.set_int(0, 1)?;
    ip.set_int(2, 3)?;
    let w = Mat::from_f32(&[2., 3., 4.], &[3])?;
    let weights = ModelBin::from_mats(&[&w])?;
    unsafe {
        weighted.load_param(&ip)?;
        weighted.load_model(weights.as_raw())?;
        drop(weights);
        drop(w);
        weighted.create_pipeline()?;
        assert_eq!(
            weighted
                .forward(&Mat::from_f32(&[1., 2., 3.], &[3])?)?
                .to_f32()?,
            vec![20.]
        );
    }
    let mut quantized = Net::new()?;
    let mut opt = Options::new()?;
    opt.set(CpuOption::Threads, 1)?;
    quantized.configure(&opt)?;
    quantized.load_param(
        "7767517\n2 2\nInput input 0 1 input\nInnerProduct fc 1 1 input output 0=1 2=3 8=1\n",
    )?;
    let mut bytes = 0x000D4B38u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[2, 3, 4, 0]);
    bytes.extend_from_slice(&1f32.to_le_bytes());
    bytes.extend_from_slice(&1f32.to_le_bytes());
    quantized.load_model(&bytes)?;
    let mut ex = quantized.extractor()?;
    ex.input("input", Mat::from_f32(&[1., 2., 3.], &[3])?)?;
    assert_eq!(ex.extract_owned("output")?.values, vec![20.]);
    let reader = ncnn::DataReader::from_bytes(&[1, 2, 3])?;
    unsafe {
        let raw = reader.as_raw();
        let mut bytes = [0u8; 4];
        assert_eq!(
            (*raw).read.unwrap()(raw, bytes.as_mut_ptr().cast(), 4),
            3,
            "a read is clamped to the bytes that remain"
        );
        assert_eq!((*raw).read.unwrap()(raw, bytes.as_mut_ptr().cast(), 1), 0);
        assert_eq!(&bytes[..3], &[1, 2, 3]);
        let text = ncnn::DataReader::from_bytes(b"12 34")?;
        let mut number = 0i32;
        assert_eq!(
            (*text.as_raw()).scan.unwrap()(
                text.as_raw(),
                c"%d".as_ptr(),
                (&mut number as *mut i32).cast()
            ),
            1,
            "the reader scans parameter text"
        );
        assert_eq!(number, 12);
    }
    let destroyed = std::sync::atomic::AtomicUsize::new(0);
    {
        let mut custom = Net::new()?;
        unsafe {
            custom.register_custom_layer(
                "RustCopy",
                Some(create_copy),
                Some(destroy_copy),
                (&destroyed as *const std::sync::atomic::AtomicUsize)
                    .cast_mut()
                    .cast(),
            )?;
        }
        custom
            .load_param("7767517\n2 2\nInput input 0 1 input\nRustCopy copy 1 1 input output\n")?;
        custom.load_model(&[])?;
        let mut ex = custom.extractor()?;
        ex.input("input", a.deep_clone()?)?;
        assert!(ex.extract("output").is_err());
        assert_eq!(ex.extract_owned("output")?.values, a.to_f32()?);
    }
    assert_eq!(destroyed.load(std::sync::atomic::Ordering::Relaxed), 1);
    unsafe {
        let mut custom = Net::new()?;
        custom.register_custom_layer(
            "RustCopy",
            Some(create_copy),
            Some(destroy_copy),
            (&destroyed as *const std::sync::atomic::AtomicUsize)
                .cast_mut()
                .cast(),
        )?;
        custom
            .load_param("7767517\n2 2\nInput input 0 1 input\nRustCopy copy 1 1 input output\n")?;
        custom.load_model(&[])?;
        let raw = sys::ncnn_mat_create_1d_batch(2, 2, std::ptr::null_mut());
        for batch in 0..2 {
            let data = sys::ncnn_mat_get_batch_data(raw, batch).cast::<f32>();
            *data = batch as f32;
            *data.add(1) = 3.;
        }
        let mat = Mat::from_raw_owned(raw, true)?;
        let mut ex = custom.extractor()?;
        ex.input("input", mat.deep_clone()?)?;
        assert_eq!(ex.extract_owned("output")?.info().n, 2);
    }
    assert_eq!(destroyed.load(std::sync::atomic::Ordering::Relaxed), 2);
    callback_lifetimes()?;
    advanced::run()?;
    ergonomic_paths()?;
    low_copy_outputs()?;
    Ok(())
}

fn low_copy_outputs() -> ncnn::Result<()> {
    let graph = "7767517\n4 5\nInput input 0 1 input\nSplit split 1 2 input a b\nReLU relu 1 1 a positive\nAbsVal abs 1 1 b absolute\n";
    for light in [0, 1] {
        for pool in [0, 1] {
            for packing in [0, 1] {
                let net = Net::builder()?
                    .threads(1)?
                    .option(CpuOption::LightMode, light)?
                    .option(CpuOption::LocalPool, pool)?
                    .option(CpuOption::Packing, packing)?
                    .load_bytes(graph, &[])?;
                let mut ex = net.extractor()?;
                let values: Vec<f32> = (0..32)
                    .map(|i| {
                        if i % 2 == 0 {
                            -(i as f32) - 2.
                        } else {
                            i as f32 + 2.
                        }
                    })
                    .collect();
                let positive: Vec<f32> = values.iter().map(|v| v.max(0.)).collect();
                let absolute: Vec<f32> = values.iter().map(|v| v.abs()).collect();
                let input = Mat::from_f32(&values, &[1, 2, 16])?;
                let pointer = input.channel_f32(0)?.as_ptr();
                ex.input("input", input)?;
                {
                    let view = ex.extract(0)?;
                    assert_eq!(view.channel_f32(0)?.as_ptr(), pointer);
                    assert_eq!(view.channel_f32(1)?, [-4., 5.]);
                    assert!(view.as_f32_slice().is_err());
                    assert!(view.channel_f32(16).is_err());
                    assert_eq!(view.info().elempack, 1);
                }
                ex.input("input", Mat::from_f32(&values, &[1, 2, 16])?)?;
                let input_copy = ex.extract("input")?.detach()?;
                assert!(ex.extract("missing").is_err());
                assert!(ex.extract("invalid\0name").is_err());
                assert!(ex.extract_owned(-1).is_err());
                assert!(ex.extract_owned(i32::MAX).is_err());
                for _ in 0..3 {
                    let view = ex.extract("positive")?;
                    assert_eq!(view.to_f32()?, positive);
                }
                assert_eq!(ex.extract("absolute")?.to_f32()?, absolute);
                {
                    let view = ex.extract("positive")?;
                    let channel = view.channel_f32(0)?;
                    assert!(net.extractor().is_err());
                    assert_eq!(channel, &positive[..2]);
                    assert_eq!(view.to_f32()?, positive);
                }
                {
                    let other = net.extractor()?;
                    assert!(ex.extract("positive").is_err());
                    drop(other);
                }
                let detached = ex.extract("positive")?.detach()?;
                let absolute_copy = ex.extract_owned("absolute")?;
                drop(ex);
                drop(net);
                assert_eq!(detached.to_f32()?, positive);
                assert_eq!(input_copy.to_f32()?, values);
                assert_eq!(absolute_copy.values, absolute);
            }
        }
    }
    let net = Net::builder()?.threads(1)?.load_bytes(graph, &[])?;
    let mut ex = net.extractor()?;
    let mut external = [1f32, -2., 3., -4.];
    let adopted = unsafe {
        Mat::from_raw_owned(
            sys::ncnn_mat_create_external_1d(4, external.as_mut_ptr().cast(), std::ptr::null_mut()),
            true,
        )?
    };
    ex.input("input", adopted)?;
    {
        let view = ex.extract("input")?;
        assert_ne!(view.as_f32_slice()?.as_ptr(), external.as_ptr());
        external.fill(99.);
        assert_eq!(view.as_f32_slice()?, [1., -2., 3., -4.]);
    }
    unsafe { ex.as_raw() };
    assert!(ex.extract("positive").is_err());
    assert_eq!(ex.extract_owned("positive")?.values, [1., 0., 3., 0.]);
    drop(ex);
    {
        let mut batched = net.extractor()?;
        let input = unsafe {
            let raw = sys::ncnn_mat_create_1d_batch(2, 2, std::ptr::null_mut());
            assert!(!raw.is_null());
            for batch in 0..2 {
                let values = sys::ncnn_mat_get_batch_data(raw, batch).cast::<f32>();
                values.write(1.);
                values.add(1).write(2.);
            }
            Mat::from_raw_owned(raw, true)?
        };
        batched.input("input", input)?;
        let view = batched.extract("input")?;
        assert_eq!(view.info().n, 2);
        assert!(view.as_f32_slice().is_err());
        assert!(view.channel_f32(0).is_err());
        assert_eq!(view.to_f32()?, [1., 2., 1., 2.]);
    }
    let mut ex = net.extractor()?;
    unsafe { net.as_raw() };
    assert!(matches!(
        ex.extract("positive"),
        Err(ncnn::Error::Invalid(_))
    ));
    let forgotten = Net::builder()?.threads(1)?.load_bytes(graph, &[])?;
    let mut cached = forgotten.extractor()?;
    cached.input("input", Mat::from_f32(&[-1., 2.], &[2])?)?;
    std::mem::forget(cached.extract("input")?);
    assert!(forgotten.extractor().is_err());
    assert!(cached
        .input("input", Mat::from_f32(&[3., 4.], &[2])?)
        .is_err());
    assert!(cached
        .input("input", Mat::from_f32(&[3., 4.], &[2])?)
        .is_err());
    assert!(cached.input(0, Mat::from_f32(&[3., 4.], &[2])?).is_err());
    assert!(cached.extract_owned("positive").is_err());
    assert!(cached.extract_into("positive", &mut Vec::new()).is_err());
    drop(cached);
    assert!(forgotten.extractor().is_err());
    let forgotten_extractor = Net::builder()?.threads(1)?.load_bytes(graph, &[])?;
    std::mem::forget(forgotten_extractor.extractor()?);
    let mut peer = forgotten_extractor.extractor()?;
    assert!(peer.extract("input").is_err());
    let mut native = Net::new()?;
    native.load_param(graph)?;
    let _ =
        unsafe { native.load_model_file_native(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md")) };
    native.load_param(graph)?;
    native.load_model(&[])?;
    assert!(matches!(
        native.extractor()?.extract("input"),
        Err(ncnn::Error::Invalid(_))
    ));
    let options = Options::new()?;
    let pool = PoolAllocator::new()?;
    unsafe {
        sys::ncnn_option_set_blob_allocator(options.as_raw(), pool.as_raw());
    }
    let mut net = Net::new()?;
    net.configure(&options)?;
    net.load_param(graph)?;
    net.load_model(&[])?;
    let mut ex = net.extractor()?;
    ex.input("input", Mat::from_f32(&[-1., 2.], &[2])?)?;
    assert!(ex.extract(0).is_err());
    assert_eq!(ex.extract_owned("positive")?.values, [0., 2.]);
    drop(ex);
    drop(net);
    drop(pool);
    Ok(())
}

fn ergonomic_paths() -> ncnn::Result<()> {
    use ncnn::image::PixelImage;
    assert!(Net::builder()?.threads(0).is_err());
    let net = Net::builder()?.threads(1)?.load_bytes(GRAPH, &[])?;
    let a = Mat::from_f32(&[1., 2., 3., 4.], &[4])?;
    let b = Mat::from_f32(&[2., 3., 4., 5.], &[4])?;
    let outputs = net.infer(
        [("a", a.deep_clone()?), ("b", b.deep_clone()?)],
        &["sum", "product"],
    )?;
    assert_eq!(outputs[0].values, [3., 5., 7., 9.]);
    assert_eq!(outputs[1].values, [2., 6., 12., 20.]);
    assert!(net
        .infer(
            [("a", a.deep_clone()?), ("b", b.deep_clone()?)],
            &["sum", "missing"]
        )
        .is_err());
    let mut sequential = net.extractor()?;
    sequential.inputs([("a", a.deep_clone()?), ("b", b.deep_clone()?)])?;
    let mut seen = Vec::new();
    sequential.outputs_with(&["sum", "product"], |name, output| {
        seen.push((name.to_owned(), output.to_f32()?));
        Ok(())
    })?;
    assert_eq!(seen[0], ("sum".to_owned(), vec![3., 5., 7., 9.]));
    assert_eq!(seen[1], ("product".to_owned(), vec![2., 6., 12., 20.]));
    let mut calls = 0;
    assert!(sequential
        .outputs_with(&["sum", "product"], |_, _| {
            calls += 1;
            Err(ncnn::Error::Invalid("consumer stopped".into()))
        })
        .is_err());
    assert_eq!(calls, 1);
    assert_eq!(sequential.extract("product")?.to_f32()?, seen[1].1);
    assert!(sequential
        .inputs([("a", a.deep_clone()?), ("bad\0name", b.deep_clone()?)])
        .is_err());
    assert_eq!(sequential.extract("a")?.to_f32()?, a.to_f32()?);
    let dense = a.as_f32_slice()?;
    assert_eq!(dense.as_ptr(), unsafe {
        sys::ncnn_mat_get_data(a.as_raw()).cast::<f32>()
    });
    let mut ex = net.extractor()?;
    let pointer = unsafe { sys::ncnn_mat_get_data(a.as_raw()) };
    ex.input("a", a)?;
    unsafe {
        let mut raw = std::ptr::null_mut();
        assert_eq!(sys::ncnn_rs_extract(ex.as_raw(), 0, 1, &mut raw), 0);
        assert_eq!(sys::ncnn_mat_get_data(raw), pointer);
        sys::ncnn_mat_destroy(raw);
    }
    ex.input("b", b.deep_clone()?)?;
    assert_eq!(ex.extract_owned("sum")?.values, [3., 5., 7., 9.]);
    let mut external = vec![1f32, 2., 3., 4.];
    let original = external.as_mut_ptr();
    let adopted = unsafe {
        Mat::from_raw_owned(
            sys::ncnn_mat_create_external_1d(4, original.cast(), std::ptr::null_mut()),
            true,
        )?
    };
    let mut ex = net.extractor()?;
    ex.input("a", adopted)?;
    unsafe {
        let mut raw = std::ptr::null_mut();
        assert_eq!(sys::ncnn_rs_extract(ex.as_raw(), 0, 1, &mut raw), 0);
        assert_ne!(sys::ncnn_mat_get_data(raw), original.cast());
        sys::ncnn_mat_destroy(raw);
    }
    drop(external);
    ex.input("b", b.deep_clone()?)?;
    assert_eq!(ex.extract_owned("sum")?.values, [3., 5., 7., 9.]);
    let pixels = [
        10, 20, 30, 40, 50, 60, 165, 165, 70, 80, 90, 100, 110, 120, 165, 165,
    ];
    let image = PixelImage::with_stride(&pixels, (2, 2), 8, PixelFormat::Rgb)?;
    let converted = image.to_mat(PixelFormat::Bgr)?;
    assert_eq!(
        converted.to_f32()?,
        [30., 60., 90., 120., 20., 50., 80., 110., 10., 40., 70., 100.]
    );
    let cropped = image.roi((1, 0), (1, 2))?.to_mat(PixelFormat::Rgb)?;
    assert_eq!(cropped.channel_f32(1)?, [50., 110.]);
    assert!(cropped.as_f32_slice().is_err());
    assert!(cropped.channel_f32(3).is_err());
    assert!(image.roi((2, 0), (1, 1)).is_err());
    assert!(PixelImage::with_stride(&pixels[..5], (2, 2), 8, PixelFormat::Rgb).is_err());
    let solid = [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30];
    let prepared = PixelImage::new(&solid, (2, 2), PixelFormat::Bgr)?
        .resize_to_mat(PixelFormat::Rgb, (1, 1))?
        .normalized(&[10., 10., 10.], &[0.5; 3])?;
    assert_eq!(prepared.to_f32()?, [10., 5., 0.]);
    Ok(())
}

unsafe extern "C" fn create_copy(_: *mut std::ffi::c_void) -> sys::ncnn_layer_t {
    let layer = sys::ncnn_layer_create();
    sys::ncnn_layer_set_one_blob_only(layer, 1);
    sys::ncnn_rs_layer_flag(layer, 1, 1, 1);
    (*layer).forward_1 = Some(copy_forward);
    layer
}
unsafe extern "C" fn destroy_copy(layer: sys::ncnn_layer_t, userdata: *mut std::ffi::c_void) {
    sys::ncnn_layer_destroy(layer);
    let count = &*userdata.cast::<std::sync::atomic::AtomicUsize>();
    count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
unsafe extern "C" fn copy_forward(
    _: sys::ncnn_layer_t,
    input: sys::ncnn_mat_t,
    out: *mut sys::ncnn_mat_t,
    _: sys::ncnn_option_t,
) -> i32 {
    std::panic::catch_unwind(|| sys::ncnn_rs_mat_clone(input, out)).unwrap_or(-1)
}

static PIPELINE_CLEANUPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn callback_lifetimes() -> ncnn::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    type Creator = unsafe extern "C" fn(*mut std::ffi::c_void) -> sys::ncnn_layer_t;
    let destroyed = AtomicUsize::new(0);
    let context = (&destroyed as *const AtomicUsize).cast_mut().cast();
    let input = Mat::from_f32(&[2., 5.], &[2])?;
    let graph = "7767517\n2 2\nInput input 0 1 input\nCallback callback 1 1 input output\n";
    PIPELINE_CLEANUPS.store(0, Ordering::Relaxed);
    for (creator, expected) in [
        (create_failing as Creator, -37),
        (create_panicking as Creator, -39),
        (create_pipeline_failure as Creator, -40),
    ] {
        let mut net = Net::new()?;
        unsafe {
            net.register_custom_layer("Callback", Some(creator), Some(destroy_copy), context)?;
        }
        net.load_param(graph)?;
        if expected == -40 {
            assert_eq!(native_code(net.load_model(&[])), Some(-1));
        } else {
            net.load_model(&[])?;
            let mut ex = net.extractor()?;
            ex.input("input", input.deep_clone()?)?;
            assert_eq!(native_code(ex.extract_owned("output")), Some(expected));
        }
    }
    assert_eq!(destroyed.load(Ordering::Relaxed), 3);
    assert_eq!(PIPELINE_CLEANUPS.load(Ordering::Relaxed), 1);
    {
        let mut layer = Layer::new("ReLU", Options::new()?)?;
        let mut parameters = ParamDict::new()?;
        parameters.set_float(0, 0.)?;
        let weights = ModelBin::from_mats(&[])?;
        unsafe {
            layer.load_param(&parameters)?;
            layer.load_model(weights.as_raw())?;
            (*layer.as_raw()).create_pipeline = Some(fail_pipeline);
            (*layer.as_raw()).destroy_pipeline = Some(count_pipeline);
            assert_eq!(native_code(layer.create_pipeline()), Some(-40));
        }
    }
    assert_eq!(PIPELINE_CLEANUPS.load(Ordering::Relaxed), 2);
    {
        let mut net = Net::new()?;
        unsafe {
            net.register_custom_layer(
                "Partial",
                Some(create_partial),
                Some(destroy_copy),
                context,
            )?;
        }
        net.load_param(
            "7767517\n3 4\nInput a 0 1 a\nInput b 0 1 b\nPartial partial 2 2 a b x y\n",
        )?;
        net.load_model(&[])?;
        let mut ex = net.extractor()?;
        ex.input("a", input.deep_clone()?)?;
        ex.input("b", input.deep_clone()?)?;
        assert_eq!(native_code(ex.extract_owned("x")), Some(-38));
    }
    assert_eq!(destroyed.load(Ordering::Relaxed), 4);
    {
        let mut net = Net::new()?;
        unsafe {
            net.register_custom_layer("Callback", Some(create_copy), Some(destroy_copy), context)?;
        }
        assert!(net.load_param("7767517\n3 3\nInput input 0 1 input\nCallback first 1 1 input intermediate\nMissingLayer second 1 1 intermediate output\n").is_err());
    }
    assert_eq!(destroyed.load(Ordering::Relaxed), 5);
    let detached = {
        let mut net = Net::new()?;
        unsafe {
            net.register_custom_layer("Callback", Some(create_copy), Some(destroy_copy), context)?;
        }
        net.load_param("7767517\n3 3\nInput input 0 1 input\nCallback first 1 1 input intermediate\nCallback second 1 1 intermediate output\n")?;
        net.load_model(&[])?;
        let result = {
            let mut ex = net.extractor()?;
            ex.input("input", input.deep_clone()?)?;
            unsafe {
                let clone = sys::ncnn_rs_extractor_clone(ex.as_raw());
                assert!(!clone.is_null());
                let mut raw = std::ptr::null_mut();
                assert_eq!(sys::ncnn_rs_extract(clone, 2, 0, &mut raw), 0);
                let result = Mat::from_raw_owned(raw, true)?.deep_clone()?;
                sys::ncnn_extractor_destroy(clone);
                result
            }
        };
        net.clear();
        result
    };
    assert_eq!(destroyed.load(Ordering::Relaxed), 7);
    assert_eq!(detached.to_f32()?, input.to_f32()?);
    let workers: Vec<_> = (0..2)
        .map(|worker| {
            std::thread::spawn(move || -> ncnn::Result<Vec<f32>> {
                let destroyed = AtomicUsize::new(0);
                let context = (&destroyed as *const AtomicUsize).cast_mut().cast();
                let output = {
                    let mut net = Net::new()?;
                    unsafe {
                        net.register_custom_layer(
                            "Callback",
                            Some(create_copy),
                            Some(destroy_copy),
                            context,
                        )?;
                    }
                    net.load_param(graph)?;
                    net.load_model(&[])?;
                    let mut ex = net.extractor()?;
                    ex.input("input", Mat::from_f32(&[worker as f32, 5.], &[2])?)?;
                    ex.extract_owned("output")?.values
                };
                assert_eq!(destroyed.load(Ordering::Relaxed), 1);
                Ok(output)
            })
        })
        .collect();
    for (worker, thread) in workers.into_iter().enumerate() {
        assert_eq!(
            thread.join().expect("Callback worker completed")?,
            vec![worker as f32, 5.]
        );
    }
    Ok(())
}

unsafe extern "C" fn fail_forward(
    _: sys::ncnn_layer_t,
    _: sys::ncnn_mat_t,
    _: *mut sys::ncnn_mat_t,
    _: sys::ncnn_option_t,
) -> i32 {
    -37
}
unsafe extern "C" fn panic_forward(
    _: sys::ncnn_layer_t,
    _: sys::ncnn_mat_t,
    _: *mut sys::ncnn_mat_t,
    _: sys::ncnn_option_t,
) -> i32 {
    std::panic::catch_unwind(|| panic!("Contained custom-layer panic")).unwrap_or(-39)
}
unsafe extern "C" fn partial_forward(
    _: sys::ncnn_layer_t,
    input: *const sys::ncnn_mat_t,
    _: i32,
    output: *mut sys::ncnn_mat_t,
    _: i32,
    _: sys::ncnn_option_t,
) -> i32 {
    let result = sys::ncnn_rs_mat_clone(*input, output);
    if result != 0 {
        return result;
    }
    -38
}
unsafe extern "C" fn fail_pipeline(_: sys::ncnn_layer_t, _: sys::ncnn_option_t) -> i32 {
    -40
}
unsafe extern "C" fn count_pipeline(_: sys::ncnn_layer_t, _: sys::ncnn_option_t) -> i32 {
    PIPELINE_CLEANUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
unsafe extern "C" fn create_failing(context: *mut std::ffi::c_void) -> sys::ncnn_layer_t {
    let layer = create_copy(context);
    (*layer).forward_1 = Some(fail_forward);
    layer
}
unsafe extern "C" fn create_panicking(context: *mut std::ffi::c_void) -> sys::ncnn_layer_t {
    let layer = create_copy(context);
    (*layer).forward_1 = Some(panic_forward);
    layer
}
unsafe extern "C" fn create_pipeline_failure(context: *mut std::ffi::c_void) -> sys::ncnn_layer_t {
    let layer = create_copy(context);
    (*layer).create_pipeline = Some(fail_pipeline);
    (*layer).destroy_pipeline = Some(count_pipeline);
    layer
}
unsafe extern "C" fn create_partial(context: *mut std::ffi::c_void) -> sys::ncnn_layer_t {
    let layer = create_copy(context);
    sys::ncnn_layer_set_one_blob_only(layer, 0);
    (*layer).forward_n = Some(partial_forward);
    layer
}

#[test]
fn resolved_endpoints_are_bound_to_the_loaded_graph() -> ncnn::Result<()> {
    let mut net = Net::new()?;
    assert!(net.resolve_blob("a").is_err());
    net.load_param(GRAPH)?;
    net.load_model(&[])?;
    let input = net.resolve_blob("a")?;
    let output = net.resolve_blob("sum")?;
    assert!(net.resolve_blob("missing").is_err());
    let other = Net::builder()?.load_bytes(GRAPH, &[])?;
    {
        let mut ex = other.extractor()?;
        assert!(ex.input(&input, Mat::from_f32(&[1.], &[1])?).is_err());
        assert!(ex.extract_into(&output, &mut Vec::new()).is_err());
    }
    {
        let mut ex = net.extractor()?;
        assert!(ex.input(&output, Mat::from_f32(&[1.], &[1])?).is_err());
        assert!(ex.extract_into(&input, &mut Vec::new()).is_err());
        assert!(ex.extract(&input).is_err());
    }
    {
        let mut ex = net.extractor()?;
        ex.input(&input, Mat::from_f32(&[2.], &[1])?)?;
        ex.input("b", Mat::from_f32(&[3.], &[1])?)?;
        let mut values = Vec::new();
        ex.extract_into(&output, &mut values)?;
        assert_eq!(values, [5.]);
        ex.extract_append(&output, &mut values)?;
        ex.extract_append(output.index(), &mut values)?;
        assert_eq!(values, [5., 5., 5.]);
        assert!(ex.extract_append(-1, &mut values).is_err());
        assert_eq!(values, [5., 5., 5.]);
    }
    net.clear();
    assert!(net.resolve_blob("a").is_err());
    net.load_param(GRAPH)?;
    net.load_model(&[])?;
    let mut ex = net.extractor()?;
    assert!(ex.input(&input, Mat::from_f32(&[1.], &[1])?).is_err());
    assert!(ex.extract_into(&output, &mut Vec::new()).is_err());
    Ok(())
}
