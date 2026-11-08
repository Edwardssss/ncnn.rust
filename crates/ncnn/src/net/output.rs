//! Validated final outputs and their buffer boundaries.

#[derive(Debug)]
/// One requested output with independent storage.
pub struct FinalOutput {
    /// Shape and stride metadata of the extracted tensor.
    pub info: crate::TensorInfo,
    /// Logical FP32 elements without channel padding.
    pub values: Vec<f32>,
}
impl FinalOutput {
    /// Returns the tensor metadata.
    pub fn info(&self) -> crate::TensorInfo {
        self.info
    }
}
/// Shape and boundary of one named output in a concatenated final buffer.
#[derive(Debug)]
pub struct OutputRange {
    /// Blob name in request order.
    pub name: String,
    /// Shape and stride metadata of this output.
    pub info: crate::TensorInfo,
    /// Index range inside the concatenated buffer.
    pub values: std::ops::Range<usize>,
}
