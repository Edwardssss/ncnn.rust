//! Public API paths that the operator families do not reach: file loading, custom
//! readers, parameter dictionaries, unchecked tensor reads, reshape views, the three
//! endpoint forms and concatenated outputs.

use ncnn_rust::{DataReader, Endpoint, Mat, Net, ParamDict};
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
    params.set_int(0, 7).unwrap();
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
