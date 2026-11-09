//! A crate that depends on the packaged archives instead of the workspace. It runs in
//! CI to prove that the published manifest, the packaged file list and the runtime
//! library lookup work outside the repository.

use ncnn_rust::{Mat, Net};

fn main() -> ncnn_rust::Result<()> {
    let net = Net::builder()?.threads(1)?.load_bytes(
        "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n",
        &[],
    )?;
    let outputs = net.infer(
        [
            ("a", Mat::from_f32(&[1., 2., 3.], &[3])?),
            ("b", Mat::from_f32(&[4., 5., 6.], &[3])?),
        ],
        &["sum"],
    )?;
    assert_eq!(outputs[0].values, [5., 7., 9.]);
    println!("packaged crate build passed");
    Ok(())
}