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
