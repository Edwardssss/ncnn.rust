//! A crate that depends on the packaged archives instead of the workspace. It runs in
//! CI to prove that the published manifest, the packaged file list and the runtime
//! library lookup work outside the repository.
//!
//! Every request form the crate documents is reached here, so a renamed or collapsed entry
//! point breaks this build before it reaches a release.

use ncnn_rust::{
    alloc, Input, Mat, Net, NetBuilder, Options, ParamSource, TensorWidth, WeightSource,
};

const GRAPH: &str = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n";

fn main() -> ncnn_rust::Result<()> {
    // The two-stage loader takes a source on each side; an empty weight source names a graph
    // whose layers carry none.
    let mut net = Net::new()?;
    net.configure(&Options::new()?)?;
    net.load(ParamSource::Text(GRAPH), WeightSource::EMPTY)?;

    let outputs = net.infer(
        [
            ("a", Mat::from_f32(&[1., 2., 3.], &[3])?),
            ("b", Mat::from_f32(&[4., 5., 6.], &[3])?),
        ],
        &["sum"],
    )?;
    assert_eq!(outputs[0].values, [5., 7., 9.]);

    // A borrowed input and a caller buffer, which is the zero-copy request path.
    let left = [1.0f32, 2., 3.];
    let right = [4.0f32, 5., 6.];
    let mut buffer = Vec::new();
    net.infer_into(
        [
            ("a", Input::borrowed(&left, &[3])),
            ("b", Input::borrowed(&right, &[3])),
        ],
        &["sum"],
        &mut buffer,
    )?;
    assert_eq!(buffer, [5., 7., 9.]);

    // The pool that used to share the `Allocator` name with the device one.
    let mut pool = alloc::Pool::new()?;
    pool.configure(0.5, 4096)?;
    let mut options = Options::new()?;
    // SAFETY: `pool` outlives `pooled`, and nothing else allocates from it meanwhile.
    unsafe { options.set_blob_allocator(&pool)? };
    let pooled = NetBuilder::with_options(options).threads(1)?.load_bytes(GRAPH, &[])?;

    // An extractor request with the width parameter, next to the guard and the values.
    let mut extractor = pooled.extractor()?;
    extractor.input("a", Mat::from_f32(&[1., 2., 3.], &[3])?)?;
    extractor.input("b", Mat::from_f32(&[4., 5., 6.], &[3])?)?;
    let owned = extractor.extract_mat("sum", TensorWidth::Fp32)?;
    assert_eq!(owned.to_f32()?, [5., 7., 9.]);
    let guard = extractor.extract("sum", TensorWidth::Stored)?;
    assert_eq!(guard.as_f32_slice()?, [5., 7., 9.]);
    drop(guard);
    pool.clear();

    println!("packaged crate build passed");
    Ok(())
}
