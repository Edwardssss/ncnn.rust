//! Endpoint selection for extractor input and output.

use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BlobKind {
    Input,
    Output,
}
/// A graph-specific blob index resolved once before repeated inference.
/// The token remains invalid after graph replacement, including address reuse.
#[derive(Clone, Debug)]
pub struct BlobId {
    graph: Rc<()>,
    index: i32,
    kind: BlobKind,
}
impl BlobId {
    pub(crate) fn new(graph: Rc<()>, index: i32, kind: BlobKind) -> Self {
        Self { graph, index, kind }
    }
    pub(crate) fn kind(&self) -> BlobKind {
        self.kind
    }
    pub(crate) fn belongs_to(&self, graph: &Rc<()>) -> bool {
        Rc::ptr_eq(graph, &self.graph)
    }
    /// Returns the endpoint index for checked indexed APIs on the originating graph.
    /// Unlike this token, an integer does not retain graph identity across reloads.
    pub fn index(&self) -> i32 {
        self.index
    }
}

/// Selects a graph endpoint by name, exposed index or resolved token.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Endpoint<'a> {
    /// Looks the endpoint up by blob name.
    Name(&'a str),
    /// Addresses the native blob index directly.
    Index(i32),
    /// Reuses a token bound to the loaded graph.
    Blob(&'a BlobId),
}

impl<'a> From<&'a str> for Endpoint<'a> {
    fn from(name: &'a str) -> Self {
        Self::Name(name)
    }
}

impl From<i32> for Endpoint<'_> {
    fn from(index: i32) -> Self {
        Self::Index(index)
    }
}

impl<'a> From<&'a String> for Endpoint<'a> {
    fn from(name: &'a String) -> Self {
        Self::Name(name)
    }
}

impl<'a> From<&'a BlobId> for Endpoint<'a> {
    fn from(blob: &'a BlobId) -> Self {
        Self::Blob(blob)
    }
}
