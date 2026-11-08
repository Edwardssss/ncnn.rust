use ncnn::{sys, Mat, Options, ParamDict, PoolAllocator};
use ncnn_rust as ncnn;

pub fn run() -> ncnn::Result<()> {
    assert!(ncnn::cpu::physical_core_count()? > 0);
    if ncnn::cpu::supports_thread_affinity() {
        let mut mask = ncnn::cpu::CpuSet::new()?;
        mask.set(0, true)?;
        assert!(mask.contains(0)?);
        assert_eq!(mask.count(), 1);
        assert!(mask.set(ncnn::cpu::CpuSet::capacity(), true).is_err());
        mask.set(ncnn::cpu::CpuSet::capacity() - 1, true)?;
        assert!(mask.contains(ncnn::cpu::CpuSet::capacity() - 1)?);
        mask.set(ncnn::cpu::CpuSet::capacity() - 1, false)?;
        assert!(ncnn::cpu::CpuSet::for_powersave(0)?.count() > 0);
        let online = ncnn::cpu::online_processors()?;
        assert!(!online.is_empty());
        assert!(ncnn::cpu::physical_core_count()? > 0);
        #[cfg(windows)]
        {
            let mut groups = std::collections::BTreeMap::new();
            for processor in &online {
                groups.entry(processor.group).or_insert(*processor);
            }
            for processor in groups.into_values() {
                std::thread::spawn(move || -> ncnn::Result<()> {
                    let before = ncnn::cpu::current_thread_affinity()?;
                    ncnn::cpu::set_current_thread_affinity(&[processor])?;
                    let mut net = ncnn::Net::new()?;
                    let mut options = Options::new()?;
                    options.set(ncnn::CpuOption::Threads, 1)?;
                    net.configure(&options)?;
                    net.load_param(
                        "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
                    )?;
                    net.load_model(&[])?;
                    let mut ex = net.extractor()?;
                    ex.input("input", Mat::from_f32(&[-2., 3.], &[2])?)?;
                    let output = ex.extract_owned("output")?.values;
                    let observed = ncnn::cpu::current_thread_affinity()?;
                    ncnn::cpu::set_current_thread_affinity(&before)?;
                    assert_eq!(observed, vec![processor]);
                    assert_eq!(output, [0., 3.]);
                    println!(
                        "Native inference executed in processor group {}",
                        processor.group
                    );
                    Ok(())
                })
                .join()
                .expect("Processor-group worker completed")?;
            }
        }
        std::thread::spawn(|| -> ncnn::Result<()> {
            let before = ncnn::cpu::current_thread_affinity()?;
            assert!(!before.is_empty());
            let selected = before[0];
            ncnn::cpu::set_current_thread_affinity(&[selected])?;
            let actual = ncnn::cpu::current_thread_affinity()?;
            ncnn::cpu::set_current_thread_affinity(&before)?;
            assert_eq!(actual, vec![selected]);
            assert_eq!(ncnn::cpu::current_thread_affinity()?, before);
            assert!(ncnn::cpu::set_current_thread_affinity(&[]).is_err());
            assert!(
                ncnn::cpu::set_current_thread_affinity(&[ncnn::cpu::ProcessorId {
                    group: u32::MAX,
                    index: u32::MAX
                }])
                .is_err()
            );
            if (selected.index as usize) < ncnn::cpu::CpuSet::capacity() {
                let mut legacy = ncnn::cpu::CpuSet::new()?;
                legacy.set(selected.index as usize, true)?;
                unsafe {
                    legacy.apply()?;
                }
                let actual = ncnn::cpu::current_thread_affinity()?;
                ncnn::cpu::set_current_thread_affinity(&before)?;
                assert_eq!(actual, vec![selected]);
            }
            Ok(())
        })
        .join()
        .expect("Affinity worker completed")?;
    } else {
        assert!(matches!(
            ncnn::cpu::online_processors(),
            Err(ncnn::Error::Unsupported(_))
        ));
        assert!(matches!(
            ncnn::cpu::set_current_thread_affinity(&[]),
            Err(ncnn::Error::Unsupported(_))
        ));
    }
    assert!(ncnn::cpu::query(ncnn::cpu::CpuQuery::LogicalCount) > 0);
    eprintln!("advanced: graphs");
    let mut detector = ncnn::Net::new()?;
    detector.load_param("7767517\n4 4\nInput loc 0 1 loc\nInput score 0 1 score\nInput prior 0 1 prior\nDetectionOutput detect 3 1 loc score prior detections 0=2 4=0.5\n")?;
    detector.load_model(&[])?;
    let mut de = detector.extractor()?;
    de.input("loc", Mat::from_f32(&[0.; 4], &[4])?)?;
    de.input("score", Mat::from_f32(&[0.1, 0.9], &[2])?)?;
    de.input(
        "prior",
        Mat::from_f32(&[0., 0., 1., 1., 0.1, 0.1, 0.2, 0.2], &[4, 2])?,
    )?;
    let detections = de.extract("detections")?.to_f32()?;
    assert_eq!(detections.len(), 6);
    assert_eq!(detections[0], 1.);
    assert!((detections[1] - 0.9).abs() < 1e-6);
    let mut sequence = ncnn::Net::new()?;
    sequence.load_param(
        "7767517\n2 2\nInput input 0 1 input\nLSTM recurrent 1 1 input output 0=1 1=4\n",
    )?;
    let mut weights = Vec::new();
    for _ in 0..3 {
        weights.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..4 {
            weights.extend_from_slice(&0f32.to_le_bytes());
        }
    }
    sequence.load_model(&weights)?;
    let mut se = sequence.extractor()?;
    se.input("input", Mat::from_f32(&[1., 2., 3.], &[1, 3])?)?;
    assert_eq!(se.extract("output")?.to_f32()?, vec![0.; 3]);
    unsafe {
        let mut calls = [0usize; 4];
        eprintln!("advanced: model callbacks");
        let model = sys::ncnn_rs_modelbin_callbacks(Some(load_callback), calls.as_mut_ptr().cast());
        assert!(!model.is_null());
        let mut loaded = Vec::new();
        for dims in 1..=4 {
            let raw = match dims {
                1 => (*model).load_1d.unwrap()(model, 2, 1),
                2 => (*model).load_2d.unwrap()(model, 2, 2, 1),
                3 => (*model).load_3d.unwrap()(model, 2, 2, 2, 1),
                _ => sys::ncnn_rs_modelbin_load4(model, 2, 2, 2, 2, 1),
            };
            loaded.push(Mat::from_raw_owned(raw, true)?);
        }
        let exhausted = sys::ncnn_rs_modelbin_load4(model, 2, 2, 2, 2, 1);
        assert!(Mat::from_raw_owned(exhausted, true).is_err());
        sys::ncnn_modelbin_destroy(model);
        assert_eq!(calls, [1, 1, 1, 2]);
        for (index, mat) in loaded.iter().enumerate() {
            assert_eq!(mat.to_f32()?, vec![(index + 1) as f32; 1 << (index + 1)]);
        }
        let graph = ncnn::Net::new()?;
        assert_eq!(sys::ncnn_rs_net_resize_graph(graph.as_raw(), 1, 0), 0);
        assert_eq!(
            sys::ncnn_rs_net_set_blob(graph.as_raw(), 0, -1, -1, [1, 2, 1, 1, 1, 1].as_ptr()),
            0
        );
        let mut producer = 0;
        let mut consumer = 0;
        let mut shape = [0; 6];
        assert_eq!(
            sys::ncnn_rs_net_blob(
                graph.as_raw(),
                0,
                &mut producer,
                &mut consumer,
                shape.as_mut_ptr()
            ),
            0
        );
        assert_eq!(shape, [1, 2, 1, 1, 1, 1]);
        assert_eq!(sys::ncnn_rs_net_resize_graph(graph.as_raw(), 0, 0), 0);
        let input_type = sys::ncnn_layer_type_to_index(c"Input".as_ptr());
        let mut binary = Vec::new();
        for v in [7767517i32, 1, 1, input_type, 0, 1, 0, -233] {
            binary.extend_from_slice(&v.to_le_bytes());
        }
        let mut net = ncnn::Net::new()?;
        net.load_param_binary(&binary)?;
        net.load_model(&[])?;
        let input = Mat::from_f32(&[7., 9.], &[2])?;
        let mut ex = net.extractor()?;
        ex.input(0, input.deep_clone()?)?;
        assert_eq!(ex.extract_owned(0)?.values, vec![7., 9.]);
        let mut raw = std::ptr::null_mut();
        assert_eq!(sys::ncnn_rs_extract(ex.as_raw(), 0, 1, &mut raw), 0);
        sys::ncnn_mat_destroy(raw);
        let memory = sys::ncnn_rs_reader_memory(b"12 34".as_ptr(), 5);
        assert!(!memory.is_null());
        let mut number = 0i32;
        assert_eq!(
            (*memory).scan.unwrap()(memory, c"%d".as_ptr(), (&mut number as *mut i32).cast()),
            1
        );
        assert_eq!(number, 12);
        let mut reference = std::ptr::null();
        assert_eq!(
            sys::ncnn_rs_reader_reference(memory, 100, &mut reference),
            0
        );
        sys::ncnn_datareader_destroy(memory);
        for key in 0..36 {
            assert!(sys::ncnn_rs_cpu_query(key) >= 0);
        }
        let original_threads = sys::ncnn_rs_runtime_get(1);
        assert_eq!(sys::ncnn_rs_runtime_set(1, 1), 0);
        assert_eq!(sys::ncnn_rs_runtime_get(1), 1);
        assert_eq!(sys::ncnn_rs_runtime_set(1, original_threads), 0);
        let original_denormals = sys::ncnn_rs_runtime_get(5);
        let mut denormal_bits = Vec::new();
        let mut denormal_modes = Vec::new();
        for mode in 0..=3 {
            assert_eq!(sys::ncnn_rs_runtime_set(5, mode), 0);
            denormal_modes.push(sys::ncnn_rs_runtime_get(5));
            let value = std::hint::black_box(f32::from_bits(1)) * std::hint::black_box(2f32);
            denormal_bits.push(value.to_bits());
        }
        assert_eq!(sys::ncnn_rs_runtime_set(5, original_denormals), 0);
        if denormal_modes == [0, 1, 2, 3] {
            assert_eq!(denormal_bits, [2, 0, 0, 0]);
        } else {
            assert_eq!(denormal_modes, [0, 0, 0, 0]);
            assert_eq!(denormal_bits, [2, 2, 2, 2]);
        }
        let mask = sys::ncnn_rs_cpuset_create();
        assert!(!mask.is_null());
        assert_eq!(sys::ncnn_rs_cpuset_update(mask, 0, 1), 0);
        assert_eq!(sys::ncnn_rs_cpuset_contains(mask, 0), 1);
        assert_eq!(sys::ncnn_rs_cpuset_count(mask), 1);
        assert_eq!(sys::ncnn_rs_cpuset_update(mask, -1, 1), -1);
        sys::ncnn_rs_cpuset_destroy(mask);
        for (value, e4m3, e5m2) in [
            (0., 0x00, 0x00),
            (-0., 0x80, 0x80),
            (1., 0x38, 0x3c),
            (-2., 0xc0, 0xc0),
        ] {
            let half = sys::ncnn_rs_f32_to_f16(value);
            assert_eq!(sys::ncnn_rs_f16_to_f32(half), value);
            assert_eq!(
                sys::ncnn_rs_bf16_to_f32(sys::ncnn_rs_f32_to_bf16(value)),
                value
            );
            for (brain, encoded) in [e4m3, e5m2].into_iter().enumerate() {
                assert_eq!(sys::ncnn_rs_f16_to_f8(half, brain as i32), encoded);
                assert_eq!(
                    sys::ncnn_rs_f16_to_f32(sys::ncnn_rs_f8_to_f16(encoded, brain as i32)),
                    value
                );
            }
        }
        let aligned = sys::ncnn_rs_fast_malloc(128);
        assert!(!aligned.is_null());
        sys::ncnn_rs_fast_free(aligned);
        let pool = PoolAllocator::new()?;
        assert_eq!(sys::ncnn_rs_pool_configure(pool.as_raw(), 0, 0.5, 8), 0);
        let m = sys::ncnn_mat_create_1d(4, pool.as_raw());
        sys::ncnn_rs_pool_clear(pool.as_raw(), 0);
        sys::ncnn_mat_destroy(m);
        sys::ncnn_rs_pool_clear(pool.as_raw(), 0);
        let shallow_source = Mat::from_f32(&[2., 4.], &[2])?;
        let mut pooled = std::ptr::null_mut();
        assert_eq!(
            sys::ncnn_rs_mat_clone_with(shallow_source.as_raw(), pool.as_raw(), &mut pooled),
            0
        );
        let mut reshaped = std::ptr::null_mut();
        assert_eq!(
            sys::ncnn_rs_mat_reshape_with(pooled, 2, [1, 2].as_ptr(), pool.as_raw(), &mut reshaped),
            0
        );
        sys::ncnn_mat_destroy(reshaped);
        sys::ncnn_mat_destroy(pooled);
        let mut shallow = std::ptr::null_mut();
        assert_eq!(
            sys::ncnn_rs_mat_shallow(shallow_source.as_raw(), &mut shallow),
            0
        );
        assert_eq!(
            sys::ncnn_mat_get_data(shallow),
            sys::ncnn_mat_get_data(shallow_source.as_raw())
        );
        sys::ncnn_mat_destroy(shallow);
        let mut metadata = std::ptr::null_mut();
        assert_eq!(
            sys::ncnn_rs_mat_shape(shallow_source.as_raw(), &mut metadata),
            0
        );
        assert_eq!(sys::ncnn_mat_get_w(metadata), 2);
        sys::ncnn_mat_destroy(metadata);
        let mut total = 0;
        let mut bits = 0;
        assert_eq!(
            sys::ncnn_rs_mat_info(shallow_source.as_raw(), &mut total, &mut bits),
            0
        );
        assert!(total >= 2);
        assert_eq!(bits, 32);
        let storage = sys::ncnn_mat_create_1d(2, std::ptr::null_mut());
        assert_eq!(sys::ncnn_rs_mat_fill(storage, 3., 0), 0);
        assert_eq!(sys::ncnn_rs_mat_fill(storage, f32::INFINITY, 1), -1);
        assert_eq!(Mat::from_raw_owned(storage, true)?.to_f32()?, vec![3., 3.]);
        let pd = ParamDict::new()?;
        let text = b"layer\0metadata";
        assert_eq!(
            sys::ncnn_rs_param_set_string(pd.as_raw(), 1, text.as_ptr(), text.len()),
            0
        );
        let mut out = vec![0; text.len()];
        assert_eq!(
            sys::ncnn_rs_param_get_string(pd.as_raw(), 1, out.as_mut_ptr(), out.len()),
            text.len()
        );
        let cloned_pd = sys::ncnn_rs_param_clone(pd.as_raw());
        assert!(!cloned_pd.is_null());
        assert_eq!(
            sys::ncnn_rs_param_get_string(cloned_pd, 1, out.as_mut_ptr(), out.len()),
            text.len()
        );
        sys::ncnn_paramdict_destroy(cloned_pd);
        assert_eq!(out, text);
        eprintln!("advanced: transforms");
        let source = Mat::from_f32(&[0., 1., -2., 4.], &[2, 2, 1])?;
        let mut opt = Options::new()?;
        opt.set(ncnn::CpuOption::Fp16Storage, 0)?;
        opt.set(ncnn::CpuOption::Fp16Packed, 0)?;
        opt.set(ncnn::CpuOption::Fp16Arithmetic, 0)?;
        for (forward, reverse) in [(3, 4), (6, 7)] {
            let mut raw = std::ptr::null_mut();
            assert_eq!(
                sys::ncnn_rs_tensor_transform(
                    source.as_raw(),
                    &mut raw,
                    forward,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    opt.as_raw()
                ),
                0
            );
            let packed = Mat::from_raw_owned(raw, false)?;
            assert_eq!(
                sys::ncnn_rs_tensor_transform(
                    packed.as_raw(),
                    &mut raw,
                    reverse,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    opt.as_raw()
                ),
                0
            );
            assert_eq!(Mat::from_raw_owned(raw, true)?.to_f32()?, source.to_f32()?);
        }
        let unpacked = Mat::from_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[8])?;
        let mut packed = std::ptr::null_mut();
        sys::ncnn_convert_packing(unpacked.as_raw(), &mut packed, 4, opt.as_raw());
        assert!(!packed.is_null());
        let mut restored = std::ptr::null_mut();
        sys::ncnn_convert_packing(packed, &mut restored, 1, opt.as_raw());
        sys::ncnn_mat_destroy(packed);
        assert_eq!(
            Mat::from_raw_owned(restored, true)?.to_f32()?,
            unpacked.to_f32()?
        );
        let bordered = sys::ncnn_mat_create();
        sys::ncnn_copy_make_border(source.as_raw(), bordered, 1, 1, 1, 1, 0, 0., opt.as_raw());
        let border = Mat::from_raw_owned(bordered, true)?;
        assert_eq!(border.info().w, 4);
        let mut flat = std::ptr::null_mut();
        sys::ncnn_flatten(source.as_raw(), &mut flat, opt.as_raw());
        let flattened = Mat::from_raw_owned(flat, true)?;
        let mut unpacked_flat = std::ptr::null_mut();
        sys::ncnn_convert_packing(flattened.as_raw(), &mut unpacked_flat, 1, opt.as_raw());
        assert_eq!(
            Mat::from_raw_owned(unpacked_flat, true)?.to_f32()?,
            source.to_f32()?
        );
        eprintln!("advanced: quantization");
        let scales = Mat::from_f32(&[2.], &[1])?;
        let mut raw_quant = std::ptr::null_mut();
        assert_eq!(
            sys::ncnn_rs_tensor_transform(
                source.as_raw(),
                &mut raw_quant,
                8,
                0,
                0,
                scales.as_raw(),
                std::ptr::null_mut(),
                opt.as_raw()
            ),
            0
        );
        let quant = Mat::from_raw_owned(raw_quant, false)?;
        assert_eq!(quant.info().elemsize, 1);
        assert_eq!(
            sys::ncnn_rs_tensor_transform(
                quant.as_raw(),
                &mut raw_quant,
                5,
                0,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                opt.as_raw()
            ),
            0
        );
        assert_eq!(
            Mat::from_raw_owned(raw_quant, true)?.to_f32()?,
            vec![0., 2., -4., 8.]
        );
        let ints = sys::ncnn_mat_create_1d(3, std::ptr::null_mut());
        std::ptr::copy_nonoverlapping(
            [1i32, 2, 3].as_ptr(),
            sys::ncnn_mat_get_data(ints).cast(),
            3,
        );
        let bias = sys::ncnn_mat_create();
        let unit = Mat::from_f32(&[1.], &[1])?;
        assert_eq!(
            sys::ncnn_rs_tensor_transform(
                ints,
                &mut raw_quant,
                9,
                0,
                0,
                scales.as_raw(),
                bias,
                opt.as_raw()
            ),
            0
        );
        assert_eq!(
            Mat::from_raw_owned(raw_quant, true)?.to_f32()?,
            vec![2., 4., 6.]
        );
        assert_eq!(
            sys::ncnn_rs_requantize(
                ints,
                &mut raw_quant,
                unit.as_raw(),
                scales.as_raw(),
                bias,
                0,
                bias,
                opt.as_raw()
            ),
            0
        );
        let requantized = Mat::from_raw_owned(raw_quant, false)?;
        assert_eq!(requantized.info().elemsize, 1);
        assert_eq!(requantized.info().elempack, 1);
        assert_eq!(requantized.info().w, 3);
        assert_eq!(
            std::slice::from_raw_parts(
                sys::ncnn_mat_get_data(requantized.as_raw()).cast::<i8>(),
                3
            ),
            &[2, 4, 6]
        );
        sys::ncnn_mat_destroy(ints);
        sys::ncnn_mat_destroy(bias);
        for op in 0..3 {
            let mut raw = std::ptr::null_mut();
            assert_eq!(
                sys::ncnn_rs_tensor_transform(
                    source.as_raw(),
                    &mut raw,
                    op,
                    3,
                    3,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    opt.as_raw()
                ),
                0
            );
            let resized = Mat::from_raw_owned(raw, true)?;
            assert_eq!(resized.to_f32()?.len(), 9);
        }
        eprintln!("advanced: views and images");
        let vector = Mat::from_f32(&[1., 2., 3., 4.], &[4])?;
        let mut raw = std::ptr::null_mut();
        assert_eq!(sys::ncnn_rs_mat_view(vector.as_raw(), 0, 1, 2, &mut raw), 0);
        assert_eq!(Mat::from_raw_owned(raw, true)?.to_f32()?, vec![2., 3.]);
        let layer = sys::ncnn_rs_layer_create(sys::ncnn_layer_type_to_index(c"ReLU".as_ptr()), 0);
        assert!(!layer.is_null());
        assert_eq!(sys::ncnn_rs_layer_flag(layer, 0, 1, 1), 1);
        assert_eq!(sys::ncnn_rs_layer_flag(layer, 1, 1, 1), 1);
        sys::ncnn_layer_destroy(layer);
        assert_eq!(sys::ncnn_rs_net_layer_count(net.as_raw()), 1);
        let borrowed = sys::ncnn_rs_net_layer(net.as_raw(), 0);
        assert!(!borrowed.is_null());
        sys::ncnn_rs_borrowed_layer_release(borrowed);
        let copied = sys::ncnn_rs_extractor_clone(ex.as_raw());
        assert!(!copied.is_null());
        sys::ncnn_rs_extractor_clear(copied);
        sys::ncnn_extractor_destroy(copied);
        let mut pixels = [0u8; 16];
        for (i, p) in pixels.iter_mut().enumerate() {
            *p = i as u8;
        }
        assert_eq!(ncnn::image::orient(&pixels, (4, 4), 1, 1)?.0, pixels);
        assert_eq!(ncnn::image::resize(&pixels, (4, 4), 1, (2, 2))?.len(), 4);
        let mut rotated = [0u8; 16];
        sys::ncnn_rs_image_kanna_rotate_c1(pixels.as_ptr(), 4, 4, rotated.as_mut_ptr(), 4, 4, 1);
        assert_eq!(rotated, pixels);
        let mut matrix = [0f32; 6];
        sys::ncnn_rs_image_get_rotation_matrix(0., 1., 0., 0., matrix.as_mut_ptr());
        assert_eq!(matrix, [1., 0., 0., -0., 1., 0.]);
        let mut resized = [0u8; 4];
        sys::ncnn_rs_image_resize_bilinear_c1(pixels.as_ptr(), 4, 4, resized.as_mut_ptr(), 2, 2);
        assert!(resized[3] > resized[0]);
        let mut padded = [0xa5u8; 24];
        for y in 0..4 {
            padded[y * 6..y * 6 + 4].copy_from_slice(&pixels[y * 4..y * 4 + 4]);
        }
        let mut padded_out = [0xa5u8; 24];
        sys::ncnn_rs_image_kanna_rotate_c1_strided(
            padded.as_ptr(),
            4,
            4,
            6,
            padded_out.as_mut_ptr(),
            4,
            4,
            6,
            1,
        );
        assert_eq!(padded_out, padded);
        let identity = [1f32, 0., 0., 0., 1., 0.];
        sys::ncnn_rs_image_warpaffine_bilinear_c1_strided(
            padded.as_ptr(),
            4,
            4,
            6,
            padded_out.as_mut_ptr(),
            4,
            4,
            6,
            identity.as_ptr(),
            0,
            0,
        );
        for y in 0..4 {
            assert_eq!(&padded_out[y * 6 + 4..y * 6 + 6], &[0xa5, 0xa5]);
        }
        sys::ncnn_rs_image_draw_rectangle_c1_strided(
            padded_out.as_mut_ptr(),
            4,
            4,
            6,
            0,
            0,
            2,
            2,
            255,
            -1,
        );
        assert_eq!(padded_out[0], 255);
        for y in 0..4 {
            assert_eq!(&padded_out[y * 6 + 4..y * 6 + 6], &[0xa5, 0xa5]);
        }
        let mut text_width = 0;
        let mut text_height = 0;
        sys::ncnn_rs_image_get_text_drawing_size(
            c"A\nB".as_ptr(),
            8,
            &mut text_width,
            &mut text_height,
        );
        assert!(text_width > 0 && text_height > 8);
        let yuv = [128u8; 24];
        let mut rgb = [0u8; 48];
        sys::ncnn_rs_image_yuv420sp2rgb(yuv.as_ptr(), 4, 4, rgb.as_mut_ptr());
        assert_eq!(rgb[0], rgb[1]);
        assert_eq!(rgb[1], rgb[2]);
    }
    Ok(())
}

unsafe extern "C" fn load_callback(
    context: *mut std::ffi::c_void,
    dims: i32,
    w: i32,
    h: i32,
    d: i32,
    c: i32,
    _: i32,
) -> sys::ncnn_mat_t {
    std::panic::catch_unwind(|| {
        let calls = &mut *context.cast::<[usize; 4]>();
        calls[dims as usize - 1] += 1;
        if calls[dims as usize - 1] > 1 {
            return std::ptr::null_mut();
        }
        let shape = match dims {
            1 => vec![w as usize],
            2 => vec![w as usize, h as usize],
            3 => vec![w as usize, h as usize, c as usize],
            _ => vec![w as usize, h as usize, d as usize, c as usize],
        };
        Mat::from_f32(&vec![dims as f32; shape.iter().product()], &shape)
            .map(Mat::into_raw)
            .unwrap_or(std::ptr::null_mut())
    })
    .unwrap_or(std::ptr::null_mut())
}
