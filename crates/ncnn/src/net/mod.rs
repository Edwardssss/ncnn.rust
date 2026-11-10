//! Network construction, configuration and model loading.

mod blob;
mod extractor;
mod graph;
mod output;

pub use blob::{BlobId, Endpoint};
pub use extractor::{Extractor, OutputGuard};
pub use graph::{BlobInfo, LayerInfo};
pub use output::{FinalOutput, OutputRange};

use crate::{
    error::{status, status_at},
    sys, Error, Mat, Options, Result,
};
use std::{
    cell::Cell,
    ffi::{CStr, CString},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

use blob::BlobKind;

/// Low-copy eligibility shared by a network and the extractors created from it.
pub(crate) struct GuardState {
    pub(crate) low_copy: Cell<bool>,
    pub(crate) extractors: Cell<usize>,
    pub(crate) output_borrowed: Cell<bool>,
}

impl GuardState {
    fn new() -> Self {
        Self {
            low_copy: Cell::new(true),
            extractors: Cell::new(0),
            output_borrowed: Cell::new(false),
        }
    }
}

/// A network configured before its first parameter load.
///
/// ```compile_fail
/// let mut net = ncnn_rust::Net::new().unwrap();
/// let mut ex = net.extractor().unwrap();
/// net.clear();
/// ex.extract("output").unwrap();
/// ```
///
/// # Thread affinity
///
/// A network, its extractors and the tensors they exchange belong to one thread. Native blob and
/// device state is thread-affine, so `Net` is neither `Send` nor `Sync`; parallel work creates one
/// network per thread and shares only the input bytes. `Arc<Net>` therefore needs one network per
/// worker rather than a shared one.
pub struct Net {
    pub(crate) raw: sys::ncnn_net_t,
    loaded: bool,
    pub(crate) ready: bool,
    custom_names: Vec<CString>,
    pub(crate) graph: Rc<()>,
    pub(crate) guard: GuardState,
    #[cfg(feature = "vulkan")]
    device: Option<crate::gpu::Device>,
    _thread: PhantomData<Rc<()>>,
}
/// Configures a network and loads its model in one expression.
///
/// Each method changes one field and returns the builder, so a rejected value surfaces before any
/// model is read. [`load_files`](Self::load_files) and [`load_bytes`](Self::load_bytes) consume the
/// builder and return a [`Net`] whose options were applied through [`Net::configure`].
///
/// [`Net::builder`] starts from [`Options::new`], which disables light mode. Use
/// [`with_options`](Self::with_options) with [`Options::native_defaults`] to keep the native preset
/// instead.
pub struct NetBuilder {
    options: Options,
    #[cfg(feature = "vulkan")]
    device: Option<crate::gpu::Device>,
}
impl NetBuilder {
    /// Selects Vulkan for the network this builder creates.
    ///
    /// The device is retained for the network's lifetime and has to be selected before parameters
    /// load, which is why the builder offers it here. ncnn keeps per-layer CPU fallback, so this
    /// selects a backend rather than guaranteeing GPU placement. FP32 output guards become
    /// unavailable on a device network; [`Extractor::extract_stored`](crate::Extractor::extract_stored)
    /// and [`Net::infer_stored`] still serve it.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(mut self, device: crate::gpu::Device) -> Self {
        self.device = Some(device);
        self
    }
    fn configured_net(self) -> Result<Net> {
        let mut net = Net::new()?;
        net.configure(&self.options)?;
        #[cfg(feature = "vulkan")]
        if let Some(device) = self.device {
            net.set_vulkan_device(device)?;
        }
        Ok(net)
    }
    /// Starts from the supplied options; subsequent flags override only their field.
    /// Use `Options::native_defaults()` explicitly for the native preset.
    pub fn with_options(options: Options) -> Self {
        Self {
            options,
            #[cfg(feature = "vulkan")]
            device: None,
        }
    }
    /// Overrides one boolean option and keeps every other field.
    pub fn flag(mut self, flag: crate::OptionFlag, enabled: bool) -> Result<Self> {
        self.options.set_flag(flag, enabled)?;
        Ok(self)
    }
    /// Overrides one integer option after validating its range.
    ///
    /// [`CpuOption::Threads`](crate::CpuOption::Threads) must be positive,
    /// [`CpuOption::FlushDenormals`](crate::CpuOption::FlushDenormals) stays within `0..=3`,
    /// [`CpuOption::OpenmpBlocktime`](crate::CpuOption::OpenmpBlocktime) and
    /// [`CpuOption::KvCacheSequenceHint`](crate::CpuOption::KvCacheSequenceHint) accept `0`, and
    /// other keys accept `0` or `1`. Anything else returns [`Error::Invalid`].
    pub fn option(mut self, key: crate::CpuOption, value: i32) -> Result<Self> {
        self.options.set(key, value)?;
        Ok(self)
    }
    /// Sets [`CpuOption::Threads`](crate::CpuOption::Threads), the native worker count.
    ///
    /// A count of zero returns [`Error::Invalid`]; one thread runs the graph without native
    /// workers, which is the setting the conformance suite uses.
    pub fn threads(self, count: usize) -> Result<Self> {
        let count = i32::try_from(count)
            .map_err(|_| Error::invalid("Thread count exceeds native range"))?;
        self.option(crate::CpuOption::Threads, count)
    }
    /// Loads text parameters and weights from files.
    ///
    /// The parameter file is the text form that begins with the `7767517` magic line. ncnn reads
    /// each layer's weights in declaration order, so the two files must come from the same model.
    /// A graph without weight layers accepts an empty weights file.
    pub fn load_files(
        self,
        parameters: impl AsRef<Path>,
        weights: impl AsRef<Path>,
    ) -> Result<Net> {
        let mut net = self.configured_net()?;
        net.load_param_file(parameters)?;
        net.load_model_file(weights)?;
        Ok(net)
    }
    /// Loads text parameters and weights from memory.
    ///
    /// `parameters` is the text form, not the binary form. Pass `&[]` for a graph whose layers
    /// carry no weights. On an existing network, [`Net::load_param`] followed by
    /// [`Net::load_model`] performs the same sequence.
    pub fn load_bytes(self, parameters: &str, weights: &[u8]) -> Result<Net> {
        let mut net = self.configured_net()?;
        net.load_param(parameters)?;
        net.load_model(weights)?;
        Ok(net)
    }
}

impl Net {
    /// Selects a Vulkan device before loading, retaining native per-layer CPU fallback.
    /// Guarded outputs are unavailable for this backend; validated buffer extraction is supported.
    #[cfg(feature = "vulkan")]
    pub fn set_vulkan_device(&mut self, device: crate::gpu::Device) -> Result<()> {
        if self.loaded {
            return Err(Error::invalid(
                "Select the backend before loading parameters",
            ));
        }
        unsafe {
            status(sys::ncnn_rs_net_vulkan(self.raw, device.index()))?;
        }
        self.device = Some(device);
        self.guard.low_copy.set(false);
        Ok(())
    }
    /// Returns the selected device, not a guarantee that every layer executes on it.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_device(&self) -> Option<&crate::gpu::Device> {
        self.device.as_ref()
    }
    /// Reports whether a device was selected, independent of the enabled features.
    pub(crate) fn has_device(&self) -> bool {
        #[cfg(feature = "vulkan")]
        {
            self.device.is_some()
        }
        #[cfg(not(feature = "vulkan"))]
        {
            false
        }
    }
    /// Starts a builder with default options.
    pub fn builder() -> Result<NetBuilder> {
        Ok(NetBuilder {
            options: Options::new()?,
            #[cfg(feature = "vulkan")]
            device: None,
        })
    }
    /// Runs owned inputs and returns validated, independently owned final buffers.
    ///
    /// Every output is converted to FP32, so a network that returns quantized storage needs
    /// [`infer_stored`](Self::infer_stored) instead.
    ///
    /// ```
    /// let net = ncnn_rust::Net::builder()?.threads(1)?.load_bytes(
    ///     "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
    ///     &[],
    /// )?;
    /// let outputs = net.infer(
    ///     [("input", ncnn_rust::Mat::from_f32(&[-1.0, 2.0], &[2])?)],
    ///     &["output"],
    /// )?;
    /// assert_eq!(outputs[0].values, [0.0, 2.0]);
    /// # Ok::<(), ncnn_rust::Error>(())
    /// ```
    pub fn infer(
        &self,
        inputs: impl IntoIterator<Item = (impl AsRef<str>, Mat)>,
        outputs: &[&str],
    ) -> Result<Vec<FinalOutput>> {
        let mut extractor = self.extractor()?;
        for (name, mat) in inputs {
            extractor.input(Endpoint::Name(name.as_ref()), mat)?;
        }
        outputs
            .iter()
            .map(|name| extractor.extract_owned(Endpoint::Name(name)))
            .collect()
    }
    /// Runs owned inputs and returns stored storage with the metadata of each output.
    ///
    /// The element width of every output is preserved, so a quantized model is readable through
    /// [`Mat::to_f32_typed`](crate::Mat::to_f32_typed) or [`Mat::to_i8`](crate::Mat::to_i8).
    pub fn infer_stored(
        &self,
        inputs: impl IntoIterator<Item = (impl AsRef<str>, Mat)>,
        outputs: &[&str],
    ) -> Result<Vec<(crate::TensorInfo, Mat)>> {
        let mut extractor = self.extractor()?;
        for (name, mat) in inputs {
            extractor.input(Endpoint::Name(name.as_ref()), mat)?;
        }
        outputs
            .iter()
            .map(|name| {
                let mat = extractor.extract_stored_owned(Endpoint::Name(name))?;
                let info = mat.info();
                Ok((info, mat))
            })
            .collect()
    }
    /// Replaces a caller-owned vector with concatenated outputs in name order.
    /// Returned metadata identifies every output range. Input failure preserves
    /// the vector; extraction failure leaves only completed output values.
    pub fn infer_into(
        &self,
        inputs: impl IntoIterator<Item = (impl AsRef<str>, Mat)>,
        outputs: &[&str],
        buffer: &mut Vec<f32>,
    ) -> Result<Vec<OutputRange>> {
        let mut extractor = self.extractor()?;
        for (name, mat) in inputs {
            extractor.input(Endpoint::Name(name.as_ref()), mat)?;
        }
        buffer.clear();
        let mut ranges = Vec::with_capacity(outputs.len());
        for name in outputs {
            let start = buffer.len();
            let info = extractor.extract_append(Endpoint::Name(name), buffer)?;
            ranges.push(OutputRange {
                name: (*name).to_owned(),
                info,
                values: start..buffer.len(),
            });
        }
        Ok(ranges)
    }
    /// Creates an empty network with default options.
    ///
    /// The network holds no graph until parameters load. [`Net::builder`] wraps this call and the
    /// configuration steps, which keeps a failure in one expression.
    pub fn new() -> Result<Self> {
        let options = Options::new()?;
        let raw = unsafe { sys::ncnn_net_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        unsafe {
            sys::ncnn_net_set_option(raw, options.raw);
        }
        Ok(Self {
            raw,
            loaded: false,
            ready: false,
            custom_names: Vec::new(),
            graph: Rc::new(()),
            guard: GuardState::new(),
            #[cfg(feature = "vulkan")]
            device: None,
            _thread: PhantomData,
        })
    }
    /// Applies an option set to a network that has not loaded parameters.
    ///
    /// The call copies the option block, so later writes to `options` do not reach this network.
    /// Vulkan compute is forced off here because device selection goes through
    /// `set_vulkan_device`, which retains the device object. A network that already selected a
    /// device keeps it. Returns [`Error::Invalid`] once parameters have loaded.
    pub fn configure(&mut self, options: &Options) -> Result<()> {
        if self.loaded {
            return Err(Error::invalid("Configure before loading parameters"));
        }
        unsafe {
            sys::ncnn_net_set_option(self.raw, options.raw);
            // Safe Vulkan selection always requires a retained Device below.
            sys::ncnn_option_set_use_vulkan_compute(sys::ncnn_net_get_option(self.raw), 0);
        }
        #[cfg(feature = "vulkan")]
        if let Some(device) = &self.device {
            unsafe {
                status(sys::ncnn_rs_net_vulkan(self.raw, device.index()))?;
            }
        }
        self.guard
            .low_copy
            .set(self.guard.low_copy.get() && options.low_copy.get());
        Ok(())
    }
    /// Replaces the graph from text parameters.
    ///
    /// The text form begins with the `7767517` magic line. Loading releases the previous layers,
    /// weights and cached blobs first, and a failure leaves the network empty instead of partially
    /// loaded, so weights load again before the next extractor. Weights load afterwards through
    /// [`load_model`](Self::load_model).
    ///
    /// Loading needs mutable access, which is what keeps live extractors and resolved endpoints out
    /// of the replaced graph.
    pub fn load_param(&mut self, text: &str) -> Result<()> {
        let text = CString::new(text)?;
        self.clear();
        let result = unsafe {
            status_at(
                "load parameters",
                sys::ncnn_net_load_param_memory(self.raw, text.as_ptr()),
            )
        };
        self.loaded = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Reads the text form from a file and applies [`load_param`](Self::load_param).
    pub fn load_param_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.load_param(&std::fs::read_to_string(path)?)
    }
    /// Loads parameters in the binary form that `ncnn2mem` writes.
    ///
    /// This path avoids parsing text at start-up. It clears the previous graph the same way
    /// [`load_param`](Self::load_param) does, and weights still load through
    /// [`load_model`](Self::load_model).
    pub fn load_param_binary(&mut self, data: &[u8]) -> Result<()> {
        self.clear();
        let result = unsafe {
            status_at(
                "load parameters",
                sys::ncnn_rs_load_bytes(self.raw, data.as_ptr(), data.len(), 0),
            )
        };
        self.loaded = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Loads weights for the current parameters, once.
    ///
    /// ncnn reads each layer's weights in declaration order, so the buffer has to match the loaded
    /// parameter text. Reload parameters before loading different weights. Detached outputs keep
    /// their own storage after a later reload; borrowed ones do not, which the mutable borrow
    /// already prevents.
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        if !self.loaded || self.ready {
            return Err(Error::invalid(
                "Load parameters before loading weights once",
            ));
        }
        let result = unsafe {
            status_at(
                "load weights",
                sys::ncnn_rs_load_bytes(self.raw, data.as_ptr(), data.len(), 1),
            )
        };
        self.ready = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Reads the weights from a file and applies [`load_model`](Self::load_model).
    pub fn load_model_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.load_model(&std::fs::read(path)?)
    }
    /// Loads weights through the native path API, including configured memory mapping.
    ///
    /// # Safety
    /// The model file's contents and length remain unchanged until the network is
    /// cleared or destroyed when mapped loading is enabled. This advanced loading
    /// path permanently disables low-copy output guards for the network.
    pub unsafe fn load_model_file_native(&mut self, path: impl AsRef<Path>) -> Result<()> {
        if !self.loaded || self.ready {
            return Err(Error::invalid(
                "Load parameters before loading weights once",
            ));
        }
        self.guard.low_copy.set(false);
        #[cfg(unix)]
        let code = {
            use std::os::unix::ffi::OsStrExt;
            let path = CString::new(path.as_ref().as_os_str().as_bytes())?;
            sys::ncnn_net_load_model(self.raw, path.as_ptr())
        };
        #[cfg(windows)]
        let code = {
            use std::os::windows::ffi::OsStrExt;
            let mut path: Vec<u16> = path.as_ref().as_os_str().encode_wide().collect();
            if path.contains(&0) {
                return Err(Error::invalid("Path contains a null character"));
            }
            path.push(0);
            sys::ncnn_net_load_model_w(self.raw, path.as_ptr())
        };
        let result = status(code);
        self.ready = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Releases every layer, weight and cached blob.
    ///
    /// Options and the device selection survive; loading parameters again rebuilds the graph and
    /// weights must load again before extraction. Existing extractors and resolved endpoints have
    /// to be dropped first, which the mutable borrow enforces, and a token from the cleared graph
    /// is rejected rather than resolving to a new blob.
    pub fn clear(&mut self) {
        self.graph = Rc::new(());
        unsafe {
            sys::ncnn_net_clear(self.raw);
        }
        self.loaded = false;
        self.ready = false;
    }
    /// Resolves an exposed graph input or output without retaining a CString.
    ///
    /// Resolution needs the loaded weights, because ncnn exposes its endpoint list only after the
    /// graph is ready. Intermediate blobs are deliberately outside this boundary: extracting one
    /// would depend on the layers between it and the outputs. The returned [`BlobId`] belongs to
    /// this graph and stops resolving after a reload.
    pub fn resolve_blob(&self, name: &str) -> Result<BlobId> {
        if !self.ready {
            return Err(Error::invalid(
                "Load the graph and weights before resolving blobs",
            ));
        }
        for output in [false, true] {
            let count = unsafe {
                if output {
                    sys::ncnn_net_get_output_count(self.raw)
                } else {
                    sys::ncnn_net_get_input_count(self.raw)
                }
            };
            for slot in 0..count {
                let ptr = unsafe {
                    if output {
                        sys::ncnn_net_get_output_name(self.raw, slot)
                    } else {
                        sys::ncnn_net_get_input_name(self.raw, slot)
                    }
                };
                if !ptr.is_null() && unsafe { CStr::from_ptr(ptr) }.to_bytes() == name.as_bytes() {
                    let index = unsafe {
                        if output {
                            sys::ncnn_net_get_output_index(self.raw, slot)
                        } else {
                            sys::ncnn_net_get_input_index(self.raw, slot)
                        }
                    };
                    if index >= 0 {
                        let kind = if output {
                            BlobKind::Output
                        } else {
                            BlobKind::Input
                        };
                        return Ok(BlobId::new(self.graph.clone(), index, kind));
                    }
                }
            }
        }
        Err(Error::invalid("Unknown graph input or output"))
    }
    /// Returns the effective value of one option on this network.
    ///
    /// The read comes from the native option block the net actually uses, so it reflects what ncnn
    /// applied rather than what a builder requested.
    pub fn option(&self, key: crate::CpuOption) -> Result<i32> {
        let option = unsafe { sys::ncnn_net_get_option(self.raw) };
        let mut value = 0;
        unsafe { status(sys::ncnn_rs_option_get(option, key as i32, &mut value))? };
        Ok(value)
    }
    /// Runs one request over borrowed FP32 inputs without copying them.
    ///
    /// Each entry is the input name, the values in channel-plane order and the shape. The borrows
    /// cover the whole request, which is what makes the zero-copy input path safe; outputs are
    /// independent values as in [`infer`](Self::infer). This is the form to use when frames already
    /// sit in caller memory.
    pub fn infer_borrowed(
        &self,
        inputs: &[(&str, &[f32], &[usize])],
        outputs: &[&str],
    ) -> Result<Vec<FinalOutput>> {
        let mut extractor = self.extractor()?;
        let mut tensors = Vec::with_capacity(inputs.len());
        for (_, values, dimensions) in inputs {
            // SAFETY: `values` outlives this function, and neither the tensor nor the extractor
            // leaves it, so every native read happens inside the borrow.
            tensors.push(unsafe { Mat::from_external(values, dimensions)? });
        }
        for ((name, _, _), tensor) in inputs.iter().zip(&tensors) {
            // SAFETY: `tensors` keeps the borrowed storage reachable until the extractions below
            // finish, and the extractor does not outlive them.
            unsafe { extractor.input_borrowed(*name, tensor)? };
        }
        let mut collected = Vec::with_capacity(outputs.len());
        for name in outputs {
            collected.push(extractor.extract_owned(*name)?);
        }
        Ok(collected)
    }
    /// Loads text parameters through a data reader.
    ///
    /// The reader lets an application load from an archive or an embedded asset without arranging a
    /// whole file. The native parser reads it sequentially, so weights need a second reader.
    pub fn load_param_reader(&mut self, reader: &crate::DataReader) -> Result<()> {
        self.clear();
        let result = unsafe {
            status_at(
                "load parameters",
                sys::ncnn_net_load_param_datareader(self.raw, reader.as_raw()),
            )
        };
        self.loaded = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Loads weights through a data reader.
    ///
    /// Requirements match [`load_model`](Self::load_model): parameters load first, and the reader
    /// has to start at the weights block because the native parser reads it sequentially.
    pub fn load_model_reader(&mut self, reader: &crate::DataReader) -> Result<()> {
        if !self.loaded || self.ready {
            return Err(Error::invalid(
                "Load parameters before loading weights once",
            ));
        }
        let result = unsafe {
            status_at(
                "load weights",
                sys::ncnn_net_load_model_datareader(self.raw, reader.as_raw()),
            )
        };
        self.ready = result.is_ok();
        if result.is_err() {
            self.clear();
        }
        result
    }
    /// Registers a native custom-layer factory for a layer type index.
    ///
    /// Models produced by custom-layer tooling can carry the index instead of the type name, and
    /// ncnn resolves those through this table. Registration must happen before parameters load, and
    /// ncnn records the factory without validating the index: a model that requests the index
    /// builds the layer, and no other model is affected.
    ///
    /// # Safety
    ///
    /// The creator must return a layer whose layout matches the index it is registered for, and
    /// both callbacks keep the native lifecycle rules that
    /// [`register_custom_layer`](Self::register_custom_layer) documents.
    pub unsafe fn register_custom_layer_by_typeindex(
        &mut self,
        typeindex: i32,
        creator: sys::ncnn_layer_creator_t,
        destroyer: sys::ncnn_layer_destroyer_t,
        userdata: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self.loaded || creator.is_none() || destroyer.is_none() {
            return Err(Error::invalid(
                "Register complete factories before loading parameters",
            ));
        }
        self.guard.low_copy.set(false);
        sys::ncnn_net_register_custom_layer_by_typeindex(
            self.raw, typeindex, creator, destroyer, userdata,
        );
        Ok(())
    }
    /// Returns the input blob names in graph order.
    ///
    /// An entry without a name is reported as an empty string.
    pub fn input_names(&self) -> Vec<String> {
        self.endpoint_names(false)
    }
    /// Returns the output blob names in graph order.
    ///
    /// An entry without a name is reported as an empty string.
    pub fn output_names(&self) -> Vec<String> {
        self.endpoint_names(true)
    }
    fn endpoint_names(&self, output: bool) -> Vec<String> {
        unsafe {
            let count = if output {
                sys::ncnn_net_get_output_count(self.raw)
            } else {
                sys::ncnn_net_get_input_count(self.raw)
            };
            (0..count)
                .map(|i| {
                    let p = if output {
                        sys::ncnn_net_get_output_name(self.raw, i)
                    } else {
                        sys::ncnn_net_get_input_name(self.raw, i)
                    };
                    if p.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                })
                .collect()
        }
    }
    /// Creates a fresh native inference cache for one request path.
    ///
    /// The native extractor caches intermediate blobs, so a changed input shape needs a new
    /// extractor rather than a re-used one. Feeding another input into an existing extractor keeps
    /// the cached downstream blobs, which is what the key/value cache tests rely on. The call needs
    /// loaded weights, and a live scoped output reserves the network, so create the next extractor
    /// after dropping the previous guard.
    pub fn extractor(&self) -> Result<Extractor<'_>> {
        if self.guard.output_borrowed.get() {
            return Err(Error::invalid("A scoped output reserves this network"));
        }
        if !self.ready {
            return Err(Error::invalid(
                "Load parameters and weights before extraction",
            ));
        }
        let raw = unsafe { sys::ncnn_extractor_create(self.raw) };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        self.guard.extractors.set(self.guard.extractors.get() + 1);
        Ok(Extractor {
            raw,
            _net: self,
            low_copy: Cell::new(true),
        })
    }
    /// Registers an advanced native custom-layer factory before loading parameters.
    ///
    /// # Safety
    /// Callbacks and userdata must remain valid until this network is destroyed.
    /// Factories return valid owned layers; callbacks contain all Rust panics.
    /// Callback outputs must initialize logical elements and preserve allocator lifetimes.
    pub unsafe fn register_custom_layer(
        &mut self,
        name: &str,
        creator: sys::ncnn_layer_creator_t,
        destroyer: sys::ncnn_layer_destroyer_t,
        userdata: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self.loaded || creator.is_none() || destroyer.is_none() {
            return Err(Error::invalid(
                "Register complete factories before loading parameters",
            ));
        }
        let name = CString::new(name)?;
        self.guard.low_copy.set(false);
        status(sys::ncnn_rs_register_layer(
            self.raw,
            name.as_ptr(),
            0,
            creator,
            destroyer,
            userdata,
        ))?;
        self.custom_names.push(name);
        Ok(())
    }
    /// Exposes advanced network access and permanently disables low-copy outputs
    /// from this network, including its existing extractors.
    ///
    /// # Safety
    /// The caller upholds native lifetime, allocator and synchronization rules and
    /// must not use the handle after this network is dropped or cleared.
    pub unsafe fn as_raw(&self) -> sys::ncnn_net_t {
        self.guard.low_copy.set(false);
        self.raw
    }
}
impl Drop for Net {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_net_destroy(self.raw);
        }
    }
}
