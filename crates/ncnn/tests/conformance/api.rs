//! Public API paths that the operator families do not reach: file loading, custom
//! readers, parameter dictionaries, unchecked tensor reads, reshape views, the three
//! endpoint forms and concatenated outputs.

use ncnn_rust::{DataReader, Endpoint, Extractor, Mat, Net, ParamDict, PoolAllocator};
use std::io::Cursor;
const PARAM: &str = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n";

fn sum_net() -> Net {
    Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap()
}

fn operands() -> (Mat, Mat) {
    (
        Mat::from_f32(&[1., 2., 3.], &[3]).unwrap(),
        Mat::from_f32(&[4., 5., 6.], &[3]).unwrap(),
    )
}

#[test]
fn pooled_networks_reuse_intermediate_memory() {
    assert!(!sum_net().reuses_blob_memory());
    let pooled = Net::builder()
        .unwrap()
        .reuse_blob_memory()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap();
    assert!(pooled.reuses_blob_memory());

    let (input_a, input_b) = operands();
    let mut extractor = pooled.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);
    drop(extractor);

    let mut buffer = Vec::new();
    for _ in 0..4 {
        let (input_a, input_b) = operands();
        let ranges = pooled
            .infer_into([("a", input_a), ("b", input_b)], &["sum"], &mut buffer)
            .unwrap();
        assert_eq!(&buffer[ranges[0].values.clone()], [5., 7., 9.]);
        let left = vec![4f32, 5., 6.];
        let right = vec![1f32, 2., 3.];
        let ranges = pooled
            .infer_borrowed_into(
                &[("a", &left, &[3]), ("b", &right, &[3])],
                &["sum"],
                &mut buffer,
            )
            .unwrap();
        assert_eq!(&buffer[ranges[0].values.clone()], [5., 7., 9.]);
    }

    // The pools outlive the network that installed them, so a network releases both on drop.
    for _ in 0..16 {
        let pooled = Net::builder()
            .unwrap()
            .reuse_blob_memory()
            .unwrap()
            .threads(1)
            .unwrap()
            .load_bytes(PARAM, &[])
            .unwrap();
        let (input_a, input_b) = operands();
        let mut extractor = pooled.extractor().unwrap();
        extractor.input("a", input_a).unwrap();
        extractor.input("b", input_b).unwrap();
        assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);
    }
}

const REUSE_PARAM: &str =
    "7767517\n3 3\nInput    in   0 1 in\nReLU     relu 1 1 in r\nBinaryOp sum  2 1 r in out 0=0\n";

fn reuse_net() -> Net {
    Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(REUSE_PARAM, &[])
        .unwrap()
}

#[test]
fn reusing_an_extractor_restarts_the_request() {
    fn run(extractor: &mut Extractor<'_>, value: f32) -> Vec<f32> {
        extractor
            .input("in", Mat::from_f32(&[value; 3], &[3]).unwrap())
            .unwrap();
        extractor.extract_owned("out").unwrap().values
    }

    let net = reuse_net();
    let mut extractor = net.extractor().unwrap();
    assert_eq!(run(&mut extractor, -1.), [-1., -1., -1.]);
    assert_eq!(run(&mut extractor, 5.), [10., 10., 10.]);

    extractor.clear().unwrap();
    assert_eq!(run(&mut extractor, -1.), [-1., -1., -1.]);

    let mut fresh = net.extractor().unwrap();
    assert_eq!(run(&mut fresh, 5.), [10., 10., 10.]);
}

#[test]
fn a_continuation_keeps_the_cached_state() {
    let net = reuse_net();
    let mut extractor = net.extractor().unwrap();
    extractor
        .input("in", Mat::from_f32(&[6.; 3], &[3]).unwrap())
        .unwrap();
    assert_eq!(
        extractor.extract_owned("out").unwrap().values,
        [12., 12., 12.]
    );

    // A continuing input leaves the blobs in place, so the graph answers from them instead of
    // recomputing; token-by-token decoding relies on that for its key/value cache.
    extractor.continue_session();
    extractor
        .input("in", Mat::from_f32(&[-2.; 3], &[3]).unwrap())
        .unwrap();
    assert_eq!(
        extractor.extract_owned("out").unwrap().values,
        [12., 12., 12.]
    );

    // The next input is a request boundary again.
    extractor
        .input("in", Mat::from_f32(&[-2.; 3], &[3]).unwrap())
        .unwrap();
    assert_eq!(
        extractor.extract_owned("out").unwrap().values,
        [-2., -2., -2.]
    );
}

#[test]
fn extractor_settings_override_the_network() {
    // A linear graph, because light mode releases each intermediate after its consumer and a graph
    // whose input feeds two layers needs light mode off.
    let net = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(
            "7767517\n2 2\nInput in 0 1 in\nReLU  relu 1 1 in out\n",
            &[],
        )
        .unwrap();
    let mut extractor = net.extractor().unwrap();
    assert!(!extractor.light_mode());
    extractor.set_light_mode(true);
    assert!(extractor.light_mode());
    extractor.set_kvcache_max_seqlen_hint(32).unwrap();

    extractor
        .input("in", Mat::from_f32(&[-1f32; 3], &[3]).unwrap())
        .unwrap();
    assert_eq!(extractor.extract_owned("out").unwrap().values, [0., 0., 0.]);

    // The handle asks for light mode, so a borrowed input stays rejected even though the network
    // option is still off.
    let values = [1f32, 2., 3.];
    let borrowed = unsafe { Mat::from_external(&values, &[3]).unwrap() };
    assert!(unsafe { extractor.input_borrowed("in", &borrowed) }.is_err());

    // Per-request pools serve the same request, and they survive the restart a second input causes.
    let mut blobs = PoolAllocator::new().unwrap();
    blobs.configure(0.5, 1024).unwrap();
    let workspace = PoolAllocator::new().unwrap();
    let kvcache = PoolAllocator::new().unwrap();
    let mut pooled = net.extractor().unwrap();
    unsafe { pooled.set_blob_allocator(&blobs) };
    unsafe { pooled.set_workspace_allocator(&workspace) };
    unsafe { pooled.set_kvcache_allocator(&kvcache) };
    pooled.set_kvcache_max_seqlen_hint(8).unwrap();
    for value in [5f32, -2.] {
        pooled
            .input("in", Mat::from_f32(&[value; 3], &[3]).unwrap())
            .unwrap();
        assert_eq!(
            pooled.extract_owned("out").unwrap().values,
            [value.max(0.); 3]
        );
    }
    blobs.clear();
}

#[test]
fn tensor_views_alias_the_source() {
    let values: Vec<f32> = (0..8).map(|index| index as f32).collect();
    let tensor = Mat::from_f32(&values, &[2, 2, 2]).unwrap();
    assert!(!tensor.is_empty());
    assert_eq!(tensor.info().shape(), [2, 2, 1, 2]);
    assert_eq!(tensor.info().total(), 8);

    let channel = tensor.channel(1).unwrap();
    assert_eq!(channel.info().shape(), [2, 2, 1, 1]);
    assert_eq!(channel.as_f32_slice().unwrap(), [4., 5., 6., 7.]);

    let channels = tensor.channel_range(0, 2).unwrap();
    assert_eq!(channels.as_f32_slice().unwrap(), values.as_slice());

    let rows = tensor.row_range(1, 1).unwrap();
    // The native helper aliases rows of the first channel plane only.
    assert_eq!(rows.info().shape(), [2, 1, 1, 1]);
    assert_eq!(rows.to_f32().unwrap(), [2., 3.]);

    let elements = tensor.column_range(1, 1).unwrap();
    // The native helper aliases a run of the first row.
    assert_eq!(elements.info().shape(), [1, 1, 1, 1]);
    assert_eq!(elements.to_f32().unwrap(), [1.0]);

    let plane = tensor.depth(0).unwrap();
    assert_eq!(plane.info().shape(), [2, 2, 1, 1]);
    assert_eq!(plane.as_f32_slice().unwrap(), [0., 1., 2., 3.]);

    let depths = tensor.depth_range(0, 1).unwrap();
    assert_eq!(depths.info().shape(), [2, 2, 1, 1]);

    assert!(tensor.channel(2).is_err());
    assert!(tensor.row_range(1, 2).is_err());
    assert!(tensor.column_range(2, 1).is_err());

    let batched = Mat::from_f32_batch(&values, &[2, 2], 2).unwrap();
    assert_eq!(
        batched.batch(0).unwrap().to_f32().unwrap(),
        [0., 1., 2., 3.]
    );
    assert_eq!(
        batched.batch(1).unwrap().to_f32().unwrap(),
        [4., 5., 6., 7.]
    );
    assert!(batched.batch(2).is_err());

    // A view copies when it goes back in as an input, because it keeps no storage of its own. The
    // graph adds the ReLU of its input to that input, so the values double.
    let net = reuse_net();
    let mut extractor = net.extractor().unwrap();
    extractor
        .input("in", tensor.channel(0).unwrap().deep_clone().unwrap())
        .unwrap();
    assert_eq!(
        extractor.extract_owned("out").unwrap().values,
        [0., 2., 4., 6.]
    );
}

#[test]
fn layer_handles_follow_the_loaded_graph() {
    let net = reuse_net();
    assert_eq!(net.layers().unwrap().len(), 3);
    let handle = unsafe { net.layer_handle(0) }.unwrap();
    assert!(!handle.is_null());
    assert!(unsafe { net.layer_handle(3) }.is_err());
}

#[test]
fn file_loading_matches_byte_loading() {
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let parameters = directory.join("api-file-loading.param");
    let weights = directory.join("api-file-loading.bin");
    std::fs::write(&parameters, PARAM).unwrap();
    std::fs::write(&weights, []).unwrap();

    let net = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_files(&parameters, &weights)
        .unwrap();
    let (input_a, input_b) = operands();
    let mut extractor = net.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);

    std::fs::remove_file(&parameters).unwrap();
    std::fs::remove_file(&weights).unwrap();
}

#[test]
fn data_reader_accepts_any_read() {
    let reader = DataReader::from_reader(Cursor::new(b"payload")).unwrap();
    assert!(!reader.as_raw().is_null());
}

#[test]
fn param_dict_round_trips_every_entry_kind() {
    let mut params = ParamDict::new().unwrap();
    assert_eq!(params.kind(0).unwrap(), None, "an unset identifier");
    params.set_int(0, 7).unwrap();
    assert_eq!(params.kind(0).unwrap(), Some(ncnn_rust::ParamKind::Int));
    params.set_float(1, 0.25).unwrap();
    assert_eq!(params.kind(1).unwrap(), Some(ncnn_rust::ParamKind::Float));
    params
        .set_array(2, &Mat::from_f32(&[1., 2.], &[2]).unwrap())
        .unwrap();
    assert_eq!(params.kind(2).unwrap(), Some(ncnn_rust::ParamKind::Array));
    params.set_float(1, 0.25).unwrap();
    let array = Mat::from_f32(&[1., 2.], &[2]).unwrap();
    params.set_array(2, &array).unwrap();
    params.set_bytes(3, b"half").unwrap();

    assert_eq!(params.get_int(0, -1).unwrap(), 7);
    assert_eq!(params.get_float(1, 0.).unwrap(), 0.25);
    assert_eq!(params.get_bytes(3).unwrap(), b"half");
    assert_eq!(params.get_int(5, -1).unwrap(), -1);
    assert_eq!(params.get_float(5, -0.5).unwrap(), -0.5);
}

#[test]
fn unchecked_read_matches_the_checked_path() {
    let mat = Mat::from_f32(&[1., 2., 3., 4.], &[2, 2]).unwrap();
    let checked = mat.to_f32().unwrap();
    let unchecked = unsafe { mat.read_f32_unchecked().unwrap() };
    assert_eq!(checked, unchecked);
}

#[test]
fn reshape_view_borrows_initialised_values() {
    let mat = Mat::from_f32(&[1., 2., 3., 4.], &[1, 2, 2]).unwrap();
    let view = mat.reshape_view(&[4]).unwrap();
    assert_eq!(view.as_f32_slice().unwrap(), [1., 2., 3., 4.]);
}

#[test]
fn endpoint_forms_address_the_same_blobs() {
    let net = sum_net();

    let mut extractor = net.extractor().unwrap();
    extractor.input("a", operands().0).unwrap();
    extractor.input("b", operands().1).unwrap();
    let by_name = extractor.extract_owned("sum").unwrap().values;

    let a = net.resolve_blob("a").unwrap();
    let b = net.resolve_blob("b").unwrap();
    let sum = net.resolve_blob("sum").unwrap();
    let mut extractor = net.extractor().unwrap();
    extractor.input(Endpoint::Blob(&a), operands().0).unwrap();
    extractor.input(Endpoint::Blob(&b), operands().1).unwrap();
    let by_blob = extractor
        .extract_owned(Endpoint::Blob(&sum))
        .unwrap()
        .values;

    let mut extractor = net.extractor().unwrap();
    extractor
        .input(Endpoint::Index(a.index()), operands().0)
        .unwrap();
    extractor
        .input(Endpoint::Index(b.index()), operands().1)
        .unwrap();
    let by_index = extractor
        .extract_owned(Endpoint::Index(sum.index()))
        .unwrap()
        .values;

    assert_eq!(by_name, [5., 7., 9.]);
    assert_eq!(by_blob, by_name);
    assert_eq!(by_index, by_name);
}

#[test]
fn concatenated_outputs_report_their_ranges() {
    let net = sum_net();
    let (input_a, input_b) = operands();
    let mut buffer = Vec::new();

    let ranges = net
        .infer_into([("a", input_a), ("b", input_b)], &["sum"], &mut buffer)
        .unwrap();

    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].name, "sum");
    assert_eq!(ranges[0].info.dims, 1);
    assert_eq!(&buffer[ranges[0].values.clone()], [5., 7., 9.]);
}

#[test]
fn effective_options_report_what_the_network_uses() {
    let net = Net::builder()
        .unwrap()
        .threads(2)
        .unwrap()
        .flag(ncnn_rust::OptionFlag::Packing, true)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap();
    assert_eq!(net.option(ncnn_rust::CpuOption::Threads).unwrap(), 2);
    assert_eq!(net.option(ncnn_rust::CpuOption::Packing).unwrap(), 1);

    let (input_a, input_b) = operands();
    let mut extractor = net.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);
    assert_eq!(
        net.option(ncnn_rust::CpuOption::Threads).unwrap(),
        2,
        "extraction keeps the network option block"
    );
}

#[test]
fn owned_tensors_support_fill_and_in_place_normalization() {
    let mut tensor = Mat::from_f32(&[0.; 6], &[3, 1, 2]).unwrap();
    tensor.fill(2.0).unwrap();
    assert_eq!(tensor.to_f32().unwrap(), [2.0; 6]);
    tensor.normalize_in_place(&[1., 1.], &[0.5, 0.25]).unwrap();
    let values = tensor.to_f32().unwrap();
    assert_eq!(values[..3], [0.5, 0.5, 0.5]);
    assert_eq!(values[3..], [0.25, 0.25, 0.25]);

    let copied = tensor.normalized(&[0.5, 0.5], &[2., 2.]).unwrap();
    assert_eq!(copied.to_f32().unwrap()[..3], [0., 0., 0.]);

    // Every coefficient combination agrees between the copying and the in-place form, and the
    // conversion is the native scale-and-bias arithmetic.
    let values = [1.0f32, -2.0, 3.5, -4.25, 5.0, -6.75];
    let source = Mat::from_f32(&values, &[3, 1, 2]).unwrap();
    for (mean, norm) in [
        (&[0.5f32, -1.0][..], &[2.0f32, 4.0][..]),
        (&[0.5f32, -1.0][..], &[][..]),
        (&[][..], &[2.0f32, 0.5][..]),
        (&[][..], &[][..]),
    ] {
        let expected: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let channel = index / 3;
                let scaled = if norm.is_empty() {
                    value
                } else {
                    value * norm[channel]
                };
                if mean.is_empty() {
                    scaled
                } else {
                    scaled - mean[channel] * norm.get(channel).copied().unwrap_or(1.0)
                }
            })
            .collect();
        let copying = source.normalized(mean, norm).unwrap().to_f32().unwrap();
        let mut in_place = source.deep_clone().unwrap();
        in_place.normalize_in_place(mean, norm).unwrap();
        let in_place = in_place.to_f32().unwrap();
        for (index, &native) in expected.iter().enumerate() {
            assert!(
                (copying[index] - native).abs() <= 1e-6,
                "copying [{index}]: {} != {native}",
                copying[index]
            );
            assert!(
                (in_place[index] - native).abs() <= 1e-6,
                "in-place [{index}]: {} != {native}",
                in_place[index]
            );
        }
    }
    let _ = tensor;
}

#[test]
fn pixels_round_trip_through_a_tensor() {
    let pixels = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let tensor = Mat::from_pixels(&pixels, ncnn_rust::PixelFormat::Rgb, 2, 2).unwrap();
    let mut out = [0u8; 12];
    tensor
        .to_pixels(ncnn_rust::PixelFormat::Rgb, 6, &mut out)
        .unwrap();
    assert_eq!(out, pixels);

    // A wider stride keeps the padding bytes untouched.
    let mut padded = [0xEEu8; 16];
    tensor
        .to_pixels(ncnn_rust::PixelFormat::Rgb, 8, &mut padded)
        .unwrap();
    assert_eq!(&padded[..6], &pixels[..6]);
    assert_eq!(&padded[6..8], &[0xEE, 0xEE]);
    assert_eq!(&padded[8..14], &pixels[6..]);

    let mut resized = [0u8; 4 * 3];
    tensor
        .to_pixels_resize(ncnn_rust::PixelFormat::Rgb, (4, 1), 12, &mut resized)
        .unwrap();
    assert!(resized.iter().any(|byte| *byte != 0));

    let short = tensor.to_pixels(ncnn_rust::PixelFormat::Rgb, 4, &mut out);
    assert!(short.is_err(), "a stride below one row is rejected");
    let wrong_format = tensor.to_pixels(ncnn_rust::PixelFormat::Gray, 6, &mut out);
    assert!(wrong_format.is_err(), "a channel mismatch is rejected");
}

#[test]
fn borders_run_on_the_native_padding_and_crop_kernels() {
    let tensor = Mat::from_f32(&[1., 2., 3., 4.], &[2, 2]).unwrap();
    let padded = tensor
        .copy_make_border(1, 1, 1, 1, ncnn_rust::BorderMode::Replicate, 0.)
        .unwrap();
    assert_eq!(padded.info().w, 4);
    assert_eq!(padded.info().h, 4);
    assert_eq!(
        padded.to_f32().unwrap(),
        [
            1., 1., 2., 2., //
            1., 1., 2., 2., //
            3., 3., 4., 4., //
            3., 3., 4., 4.,
        ]
    );

    let constant = tensor
        .copy_make_border(1, 0, 0, 1, ncnn_rust::BorderMode::Constant, -1.)
        .unwrap();
    assert_eq!(constant.info().w, 3);
    assert_eq!(constant.info().h, 3);
    assert_eq!(
        constant.to_f32().unwrap(),
        [-1., -1., -1., 1., 2., -1., 3., 4., -1.]
    );

    let cropped = padded.copy_cut_border(1, 1, 1, 1).unwrap();
    assert_eq!(cropped.info().w, 2);
    assert_eq!(cropped.to_f32().unwrap(), [1., 2., 3., 4.]);
    assert!(cropped.copy_cut_border(0, 2, 0, 0).is_err());
    let mirrored = tensor
        .copy_make_border(1, 1, 1, 1, ncnn_rust::BorderMode::Reflect, 0.)
        .unwrap();
    assert_eq!((mirrored.info().w, mirrored.info().h), (4, 4));
    let values = mirrored.to_f32().unwrap();
    let interior: Vec<f32> = [1usize, 2]
        .iter()
        .flat_map(|row| [1usize, 2].map(|column| values[row * 4 + column]))
        .collect();
    assert_eq!(interior, [1., 2., 3., 4.], "the mirror keeps the source");
}

#[test]
fn borrowed_inputs_run_without_copying() {
    let net = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(
            "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
            &[],
        )
        .unwrap();
    let values = [-1.0f32, 2.0, -3.0, 4.0];
    let outputs = net
        .infer_borrowed(&[("input", &values, &[4])], &["output"])
        .unwrap();
    assert_eq!(outputs[0].values, [0., 2., 0., 4.]);

    // The tensor keeps the caller pointer, which is what makes the path zero-copy.
    let tensor = unsafe { Mat::from_external(&values, &[4]) }.unwrap();
    assert_eq!(tensor.info().elemsize, 4);
    assert_eq!(tensor.to_f32().unwrap(), values);
    let mut extractor = net.extractor().unwrap();
    let stored = unsafe { extractor.input_borrowed("input", &tensor) };
    assert!(stored.is_ok());
    assert_eq!(
        extractor.extract_owned("output").unwrap().values,
        [0., 2., 0., 4.],
        "the native kernels read the borrowed buffer"
    );
    assert_eq!(
        tensor.to_f32().unwrap(),
        values,
        "no in-place kernel writes the caller buffer"
    );
    drop(tensor);

    // The fused form borrows the input and appends the output to a caller vector, so a
    // steady-state loop allocates on neither side.
    let mut buffer = Vec::new();
    let ranges = net
        .infer_borrowed_into(&[("input", &values, &[4])], &["output"], &mut buffer)
        .unwrap();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].name, "output");
    assert_eq!(ranges[0].info.w, 4);
    assert_eq!(&buffer[ranges[0].values.clone()], [0., 2., 0., 4.]);
    let capacity = buffer.capacity();
    net.infer_borrowed_into(&[("input", &values, &[4])], &["output"], &mut buffer)
        .unwrap();
    assert_eq!(
        buffer.capacity(),
        capacity,
        "the buffer keeps its allocation across requests"
    );
    assert!(
        net.infer_borrowed_into(&[("input", &values, &[3])], &["output"], &mut buffer)
            .is_err(),
        "the shape must match the slice"
    );

    // Light mode reaches a native refcount read that an external tensor does not carry, so the
    // combination is rejected instead of reaching that path.
    let light = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .option(ncnn_rust::CpuOption::LightMode, 1)
        .unwrap()
        .load_bytes(
            "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
            &[],
        )
        .unwrap();
    assert!(
        light
            .infer_borrowed(&[("input", &values, &[4])], &["output"])
            .is_err(),
        "light mode rejects a borrowed input"
    );
    // The owned form still runs there, because it hands over storage with a refcount.
    assert_eq!(
        light
            .infer(
                [("input", Mat::from_f32(&values, &[4]).unwrap())],
                &["output"]
            )
            .unwrap()[0]
            .values,
        [0., 2., 0., 4.]
    );

    let mismatch = unsafe { Mat::from_external(&values, &[3]) };
    assert!(mismatch.is_err(), "the shape must match the slice");
}

#[test]
fn graph_description_matches_the_parameter_text() {
    let net = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap();
    let layers = net.layers().unwrap();
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].kind, "Input");
    assert_eq!(layers[2].kind, "BinaryOp");
    assert_eq!(layers[2].name, "sum");
    assert_eq!(layers[2].bottoms, 2);
    assert_eq!(layers[2].tops, 1);
    assert!(!layers[2].one_blob_only, "BinaryOp consumes two blobs");
    assert!(
        layers[1].one_blob_only && layers[1].support_inplace,
        "ReLU is a one-blob in-place layer"
    );
    assert!(
        layers[0].typeindex > 0,
        "Input resolves through the registry"
    );
    assert!(
        layers[2].typeindex > 0,
        "BinaryOp resolves through the registry"
    );
    assert_ne!(layers[0].typeindex, layers[2].typeindex);
    assert!(
        !layers[2].support_any_packing || layers[2].support_packing,
        "any packing implies the plain packing flag"
    );
    assert!(
        !layers[2].support_vulkan_packing || layers[2].support_vulkan,
        "packed Vulkan implies plain Vulkan"
    );

    let blobs = net.blobs().unwrap();
    let names: Vec<&str> = blobs.iter().map(|blob| blob.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "sum"]);
    assert_eq!(blobs[0].producer, 0, "the first input feeds blob a");
    assert_eq!(blobs[1].producer, 1, "the second input feeds blob b");
    assert_eq!(blobs[2].producer, 2, "the third layer produces the sum");
    assert_eq!(blobs[0].consumer, 2, "the sum layer reads the first input");
    assert!(
        !blobs.iter().any(|blob| blob.has_shape()),
        "shapes appear only after inputs are sized"
    );

    // The decoded flags match the native accessors that read the same layer fields. The Vulkan
    // accessors are not comparable here: a parameter-loaded layer is built by the registry, which
    // selects the Vulkan subclass of the type, while `ncnn_layer_create_by_typeindex` builds the
    // base class and reports the plain flags.
    for layer in &layers {
        let handle = unsafe { ncnn_rust::sys::ncnn_layer_create_by_typeindex(layer.typeindex) };
        assert!(!handle.is_null(), "{} exists in the registry", layer.kind);
        unsafe {
            assert_eq!(
                ncnn_rust::sys::ncnn_layer_get_one_blob_only(handle) != 0,
                layer.one_blob_only,
                "{} one-blob flag",
                layer.kind
            );
            assert_eq!(
                ncnn_rust::sys::ncnn_layer_get_support_inplace(handle) != 0,
                layer.support_inplace,
                "{} in-place flag",
                layer.kind
            );
            assert_eq!(
                ncnn_rust::sys::ncnn_layer_get_support_packing(handle) != 0,
                layer.support_packing,
                "{} packing flag",
                layer.kind
            );
            assert_eq!(
                ncnn_rust::sys::ncnn_layer_get_support_bf16_storage(handle) != 0,
                layer.support_bf16_storage,
                "{} bf16 flag",
                layer.kind
            );
            ncnn_rust::sys::ncnn_layer_destroy(handle);
        }
    }

    // Only a network with weights reports a graph.
    let unloaded = Net::new().unwrap();
    assert!(unloaded.layers().is_err());
    assert!(unloaded.blobs().is_err());
}

#[test]
fn readers_load_parameters_and_weights() {
    let reader = DataReader::from_bytes(PARAM.as_bytes()).unwrap();
    let mut net = Net::new().unwrap();
    net.load_param_reader(&reader).unwrap();
    let weights = DataReader::from_bytes(&[]).unwrap();
    net.load_model_reader(&weights).unwrap();
    let (input_a, input_b) = operands();
    let mut extractor = net.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);

    let mut second = Net::new().unwrap();
    let empty = DataReader::from_bytes(b"").unwrap();
    assert!(second.load_param_reader(&empty).is_err());
    assert!(
        second.load_model_reader(&weights).is_err(),
        "weights need parameters first"
    );
}

#[test]
fn type_index_registration_rejects_incomplete_factories() {
    let mut net = Net::new().unwrap();
    let error =
        unsafe { net.register_custom_layer_by_typeindex(20, None, None, std::ptr::null_mut()) };
    assert!(error.is_err(), "both callbacks are required");

    let loaded = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap();
    let mut loaded = loaded;
    let late =
        unsafe { loaded.register_custom_layer_by_typeindex(20, None, None, std::ptr::null_mut()) };
    assert!(late.is_err(), "registration happens before loading");
}

#[test]
fn consuming_helpers_keep_the_storage_a_tensor_already_owns() {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Mat::from_f32(&values, &[3, 2]).unwrap();
    let reshaped = tensor.into_reshape(&[2, 3]).unwrap();
    assert_eq!(reshaped.info().dims, 2);
    assert_eq!((reshaped.info().w, reshaped.info().h), (2, 3));
    assert_eq!(reshaped.to_f32().unwrap(), values);
    let bad = Mat::from_f32(&values, &[3, 2])
        .unwrap()
        .into_reshape(&[4, 2]);
    assert!(bad.is_err(), "the element count must match");

    let normalized = Mat::from_f32(&[0.0; 6], &[3, 2])
        .unwrap()
        .into_normalized(&[1.0], &[2.0])
        .unwrap();
    assert_eq!(normalized.to_f32().unwrap(), [-2.0f32; 6]);

    // Borrowed storage is copied first, so the caller keeps its values.
    let borrowed = [4.0f32; 6];
    let external = unsafe { Mat::from_external(&borrowed, &[3, 2]) }.unwrap();
    let copied = external.into_normalized(&[1.0], &[2.0]).unwrap();
    assert_eq!(copied.to_f32().unwrap(), [6.0f32; 6]);
    assert_eq!(borrowed, [4.0f32; 6]);
}

#[test]
fn batched_tensors_run_through_a_graph() {
    let net = Net::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .load_bytes(
            "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
            &[],
        )
        .unwrap();
    let values = [-1.0f32, 2.0, -3.0, 4.0, 5.0, -6.0];
    let batched = Mat::from_f32_batch(&values, &[3], 2).unwrap();
    assert_eq!(batched.info().n, 2);
    assert_eq!(batched.to_f32().unwrap(), values);

    let mut extractor = net.extractor().unwrap();
    extractor.input("input", batched).unwrap();
    let output = extractor.extract_owned("output").unwrap();
    assert_eq!(output.info.n, 2, "the batch count survives the graph");
    assert_eq!(output.values, [0., 2., 0., 4., 5., 0.]);

    let short = Mat::from_f32_batch(&values[..5], &[3], 2);
    assert!(short.is_err(), "the data has to cover every batch");
}

#[test]
fn flatten_and_packing_rewrite_the_layout() {
    let tensor = Mat::from_f32(
        &(0..8).map(|value| value as f32).collect::<Vec<_>>(),
        &[2, 2, 2],
    )
    .unwrap();
    let flat = tensor.flatten().unwrap();
    assert_eq!(flat.info().dims, 1);
    assert_eq!(flat.info().w, 8);
    assert_eq!(
        flat.to_f32().unwrap(),
        (0..8).map(|v| v as f32).collect::<Vec<_>>()
    );

    let packed = tensor.with_packing(2).unwrap();
    assert_eq!(packed.info().elempack, 2);
    assert_eq!(packed.info().c, 1, "two channels share one lane");
    let mut lane_order = packed.to_f32_typed(ncnn_rust::ElementKind::F32).unwrap();
    let mut plane_order = tensor.to_f32().unwrap();
    lane_order.sort_by(|a, b| a.partial_cmp(b).unwrap());
    plane_order.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(lane_order, plane_order, "packing keeps every value");
    let unpacked = packed.with_packing(1).unwrap();
    assert_eq!(unpacked.info().elempack, 1);
    assert_eq!(unpacked.to_f32().unwrap(), tensor.to_f32().unwrap());
    assert!(
        tensor.with_packing(3).is_err(),
        "the packing has to divide the axis"
    );
    assert!(tensor.with_packing(0).is_err());
}

#[test]
fn depth_borders_cover_the_fourth_axis() {
    let values: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let tensor = Mat::from_f32(&values, &[2, 2, 2, 1]).unwrap();
    assert_eq!(tensor.info().dims, 4);
    let border = ncnn_rust::Borders::new(1, 0, 0, 1).with_depth(1, 0);
    let padded = tensor
        .copy_make_border_3d(border, ncnn_rust::BorderMode::Constant, -1.)
        .unwrap();
    assert_eq!(
        (
            padded.info().w,
            padded.info().h,
            padded.info().d,
            padded.info().c
        ),
        (3, 3, 3, 1)
    );
    let cropped = padded.copy_cut_border_3d(border).unwrap();
    assert_eq!(cropped.to_f32().unwrap(), values);
    assert!(
        cropped
            .copy_cut_border_3d(ncnn_rust::Borders::new(0, 0, 0, 2))
            .is_err(),
        "the crop keeps one column"
    );
    assert!(
        Mat::from_f32(&values, &[2, 2, 2])
            .unwrap()
            .copy_cut_border_3d(ncnn_rust::Borders::default())
            .is_err(),
        "the depth variant needs four axes"
    );
}

#[test]
fn readers_open_files_and_load_through_them() {
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let parameters = directory.join("api-reader-file.param");
    std::fs::write(&parameters, PARAM).unwrap();

    let reader = DataReader::from_file(&parameters).unwrap();
    let mut net = Net::new().unwrap();
    net.load_param_reader(&reader).unwrap();
    let weights = DataReader::from_bytes(&[]).unwrap();
    net.load_model_reader(&weights).unwrap();
    let (input_a, input_b) = operands();
    let mut extractor = net.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);
    assert!(
        DataReader::from_file(directory.join("api-reader-file-missing.param")).is_err(),
        "a missing file is reported"
    );
    // The reader keeps the file open, so the handle has to drop before the file can go.
    drop(reader);
    std::fs::remove_file(&parameters).unwrap();
}

#[test]
fn allocator_options_attach_pools() {
    let mut options = ncnn_rust::Options::new().unwrap();
    let blobs = ncnn_rust::PoolAllocator::new().unwrap();
    let workspace = ncnn_rust::PoolAllocator::new().unwrap();
    let cache = ncnn_rust::PoolAllocator::new().unwrap();
    unsafe {
        options.set_blob_allocator(&blobs).unwrap();
        options.set_workspace_allocator(&workspace).unwrap();
        options.set_kvcache_allocator(&cache).unwrap();
    }
    let net = ncnn_rust::NetBuilder::with_options(options)
        .threads(1)
        .unwrap()
        .load_bytes(PARAM, &[])
        .unwrap();
    let (input_a, input_b) = operands();
    let mut extractor = net.extractor().unwrap();
    extractor.input("a", input_a).unwrap();
    extractor.input("b", input_b).unwrap();
    assert_eq!(extractor.extract_owned("sum").unwrap().values, [5., 7., 9.]);
}

#[test]
fn affine_helpers_place_and_warp_pixels() {
    let from = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let to = [[0.0f32, 0.0], [2.0, 0.0], [0.0, 2.0]];
    let scale = ncnn_rust::image::affine_transform(&from, &to).unwrap();
    let inverse = ncnn_rust::image::invert_affine(&scale);
    assert!((inverse[0] - 0.5).abs() < 1e-6, "the inverse halves x");
    assert!((inverse[4] - 0.5).abs() < 1e-6, "the inverse halves y");
    assert!(ncnn_rust::image::affine_transform(&from[..1], &[]).is_err());

    let pixels = [10u8, 20, 30, 40];
    let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let warped = ncnn_rust::image::warp_affine(
        &pixels,
        (2, 2),
        1,
        (2, 2),
        &identity,
        ncnn_rust::image::WarpBorder::Replicate,
        0,
    )
    .unwrap();
    assert_eq!(warped, pixels, "an identity warp keeps the pixels");
    let blank = ncnn_rust::image::warp_affine(
        &pixels,
        (2, 2),
        1,
        (2, 2),
        &scale,
        ncnn_rust::image::WarpBorder::Constant,
        7,
    )
    .unwrap();
    assert!(blank.contains(&7), "the border fills");
}
