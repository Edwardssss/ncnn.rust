//! Pairs of layers and graphs that must agree with themselves. A round trip proves
//! layout and scale agreement without needing a second implementation as a reference.

use crate::support::{self, assert_close_with};

#[test]
fn unfold_and_fold_round_trip() {
    let params = "1=2 11=2 3=2 13=2 4=0";
    let net = support::chain(
        &[
            ("Unfold", params),
            ("Fold", "1=2 11=2 3=2 13=2 4=0 20=4 21=4"),
        ],
        &[],
    );
    let source = support::ramp(32);
    let restored = support::run_labeled(&net, &source, &[4, 4, 2], "unfold round trip");
    assert_close_with(&restored, &source, 1e-5);
}

#[test]
fn splitting_then_summing_doubles_the_input() {
    let net = support::graph(
        3,
        4,
        "Input input 0 1 input\n\
         Split split 1 2 input left right\n\
         Eltwise sum 2 1 left right output 0=1\n",
        &[],
    );
    let source = support::ramp(16);
    let doubled = support::run_labeled(&net, &source, &[4, 4], "split sum");
    let expected: Vec<f32> = source.iter().map(|value| value * 2.0).collect();
    assert_close_with(&doubled, &expected, 1e-6);
}
