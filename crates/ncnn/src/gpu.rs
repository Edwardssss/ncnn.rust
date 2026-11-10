//! Process-lifetime Vulkan initialization and explicit device selection.
//!
//! The native instance and cached devices remain alive until process exit. Raw
//! users must not destroy or replace that instance, its devices, or allocators.
use crate::{sys, Error, Result};
use std::{ffi::CStr, marker::PhantomData, rc::Rc, sync::OnceLock};

/// Storage and arithmetic policy requested from the Vulkan backend.
///
/// The variant is a request, not a guarantee: ncnn keeps its own capability dispatch, so a device
/// without half-precision support stays on FP32.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Precision {
    /// Requests FP32 storage and arithmetic.
    Fp32,
    /// Requests FP16 storage and arithmetic; native capability checks can retain FP32 on
    /// unsupported devices.
    Fp16,
}

/// An initialized native device. Clones share the process-owned device.
///
/// # Thread affinity
///
/// The native device table is thread-affine: enumerate once per thread and keep the device, its
/// allocator and every network that uses them on that thread. Devices coordinate process-owned
/// runtime state, so a device is not `Send` or `Sync`.
#[derive(Clone, Debug)]
pub struct Device {
    index: i32,
    _thread: PhantomData<Rc<()>>,
}

/// Owned metadata copied from the initialized device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Device index used for selection.
    pub index: usize,
    /// Driver-reported device name.
    pub name: String,
    /// Reported Vulkan API version.
    pub api_version: u32,
    /// PCI vendor identifier.
    pub vendor_id: u32,
    /// PCI device identifier.
    pub device_id: u32,
    /// Native device type: 0 discrete, 1 integrated, 2 virtual, 3 CPU.
    pub device_type: u32,
}

fn initialize() -> Result<()> {
    static STATUS: OnceLock<i32> = OnceLock::new();
    let code = *STATUS.get_or_init(|| unsafe { sys::ncnn_rs_gpu_initialize() });
    if code != 0 {
        return Err(Error::Unsupported("Vulkan instance initialization failed"));
    }
    Ok(())
}

impl Device {
    /// Enumerates available devices without silently substituting the CPU backend.
    ///
    /// The order is the native one and matches the index [`Device::new`] accepts. An empty vector
    /// means no driver or ICD is visible, which is a missing capability rather than an empty device
    /// list to fall back from.
    pub fn enumerate() -> Result<Vec<DeviceInfo>> {
        initialize()?;
        Ok((0..unsafe { sys::ncnn_rs_gpu_count() })
            .map(Self::metadata)
            .collect())
    }
    /// Selects one device by index after validating it.
    ///
    /// An index the driver does not expose returns [`Error::Unsupported`].
    /// The native device outlives this handle: every network that selects it keeps a reference.
    pub fn new(index: usize) -> Result<Self> {
        initialize()?;
        let index =
            i32::try_from(index).map_err(|_| Error::invalid("Invalid Vulkan device index"))?;
        if unsafe { sys::ncnn_rs_gpu_valid(index) } == 0 {
            return Err(Error::Unsupported(
                "Vulkan device is unavailable or invalid",
            ));
        }
        Ok(Self {
            index,
            _thread: PhantomData,
        })
    }
    /// Selects the device the native runtime ranks first.
    ///
    /// ncnn scores the visible devices and picks one, so this is the portable choice for an
    /// application that does not pin a GPU. Combine [`enumerate`](Self::enumerate) with
    /// [`new`](Self::new) when a specific device matters.
    pub fn default_device() -> Result<Self> {
        initialize()?;
        let index = unsafe { sys::ncnn_rs_gpu_default() };
        if index < 0 {
            return Err(Error::Unsupported("No Vulkan device is available"));
        }
        Self::new(index as usize)
    }
    /// Returns the copied device metadata.
    ///
    /// The values are read once per call, so later driver-side changes are not observed.
    pub fn info(&self) -> DeviceInfo {
        Self::metadata(self.index)
    }
    pub(crate) fn index(&self) -> i32 {
        self.index
    }
    fn metadata(index: i32) -> DeviceInfo {
        unsafe {
            DeviceInfo {
                index: index as usize,
                name: CStr::from_ptr(sys::ncnn_rs_gpu_name(index))
                    .to_string_lossy()
                    .into_owned(),
                api_version: sys::ncnn_rs_gpu_property(index, 0),
                vendor_id: sys::ncnn_rs_gpu_property(index, 1),
                device_id: sys::ncnn_rs_gpu_property(index, 2),
                device_type: sys::ncnn_rs_gpu_property(index, 3),
            }
        }
    }
}

use crate::{error::status, Mat, Net};
use std::{
    cell::Cell,
    ffi::{c_void, CString},
    path::Path,
};

/// A thread-affine allocator arena retained by all of its resident tensors.
#[derive(Clone)]
pub struct Allocator(Rc<AllocatorInner>);
struct AllocatorInner {
    raw: *mut c_void,
    device: Device,
    live_bytes: Cell<usize>,
    peak_bytes: Cell<usize>,
}
impl Drop for AllocatorInner {
    fn drop(&mut self) {
        unsafe { sys::ncnn_rs_gpu_allocator_destroy(self.raw) }
    }
}
/// A resident buffer with no unsynchronized mapped or raw access.
pub struct Tensor {
    raw: *mut c_void,
    bytes: usize,
    allocator: Allocator,
}
impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { sys::ncnn_rs_gpu_tensor_destroy(self.raw) }
        self.allocator
            .0
            .live_bytes
            .set(self.allocator.0.live_bytes.get() - self.bytes);
    }
}
impl Device {
    /// Creates a thread-affine allocator on this device.
    pub fn allocator(&self) -> Result<Allocator> {
        let raw = unsafe { sys::ncnn_rs_gpu_allocator_create(self.index) };
        if raw.is_null() {
            return Err(Error::Native(-1));
        }
        Ok(Allocator(Rc::new(AllocatorInner {
            raw,
            device: self.clone(),
            live_bytes: Cell::new(0),
            peak_bytes: Cell::new(0),
        })))
    }
    /// Reports the native device-wide heap budget in MiB, not application usage.
    pub fn heap_budget_mib(&self) -> usize {
        unsafe { sys::ncnn_rs_gpu_heap_budget(self.index) }
    }
    /// Returns the borrowed native VkDevice handle. This does not transfer ownership.
    pub fn raw_device_handle(&self) -> u64 {
        unsafe { sys::ncnn_rs_gpu_device_handle(self.index) }
    }
    /// Serializes native compatibility metadata, shaders, and Vulkan pipeline data.
    pub fn save_pipeline_cache(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut data = std::ptr::null_mut();
        let mut size = 0;
        status(unsafe { sys::ncnn_rs_gpu_cache_save(self.index, &mut data, &mut size) })?;
        let result = std::fs::write(path, unsafe { std::slice::from_raw_parts(data, size) });
        unsafe { sys::ncnn_rs_gpu_cache_free(data) };
        result?;
        Ok(())
    }
    /// Rejects incompatible or corrupted native cache headers and payload hashes.
    pub fn load_pipeline_cache(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = std::fs::read(path)?;
        status(unsafe { sys::ncnn_rs_gpu_cache_load(self.index, data.as_ptr(), data.len()) })
    }
}
impl Allocator {
    /// Reports live tensor storage and its high-water mark, excluding allocator pools, staging, workspace, and driver allocations.
    /// Imported storage is included without implying ownership.
    pub fn tensor_bytes(&self) -> (usize, usize) {
        (self.0.live_bytes.get(), self.0.peak_bytes.get())
    }
    /// Uploads and waits before returning resident FP32 storage.
    ///
    /// A device that refuses the storage reports the refusal as a native failure during
    /// `Vulkan upload`; the allocator, the tensors it already returned and later requests stay
    /// usable.
    pub fn upload(&self, mat: &Mat) -> Result<Tensor> {
        let info = mat.info();
        if info.elemsize != 4 || info.elempack != 1 || info.dims == 0 {
            return Err(Error::invalid(
                "Upload requires nonempty unpacked FP32 storage",
            ));
        }
        let raw = unsafe { sys::ncnn_rs_gpu_upload(self.0.raw, mat.as_raw()) };
        self.tensor(raw, "Vulkan upload")
    }
    fn tensor(&self, raw: *mut c_void, operation: &'static str) -> Result<Tensor> {
        if raw.is_null() {
            return Err(Error::NativeAt(operation, -1));
        }
        let bytes = unsafe { sys::ncnn_rs_gpu_tensor_bytes(raw) };
        let Some(live) = self.0.live_bytes.get().checked_add(bytes) else {
            unsafe { sys::ncnn_rs_gpu_tensor_destroy(raw) };
            return Err(Error::invalid("Tensor byte counter overflow"));
        };
        self.0.live_bytes.set(live);
        self.0.peak_bytes.set(self.0.peak_bytes.get().max(live));
        Ok(Tensor {
            raw,
            bytes,
            allocator: self.clone(),
        })
    }
    /// Records resident inference. Submission occurs only in `Command::wait`.
    pub fn command<'a>(&self, net: &'a Net) -> Result<Command<'a>> {
        if net.vulkan_device().map(Device::index) != Some(self.0.device.index) {
            return Err(Error::invalid("Network and allocator devices differ"));
        }
        let raw = unsafe { sys::ncnn_rs_gpu_command_create(self.0.raw, net.as_raw()) };
        if raw.is_null() {
            return Err(Error::NativeAt("Vulkan command creation", -1));
        }
        Ok(Command {
            raw,
            allocator: self.clone(),
            _net: net,
            inputs: Vec::new(),
            outputs: Vec::new(),
            recorded: false,
            failed: false,
        })
    }
    /// Borrows a caller-owned same-device Vulkan storage buffer as unpacked FP32.
    ///
    /// # Safety
    /// The buffer must belong to `Device::raw_device_handle`, support storage and
    /// transfer usage, and remain allocated and exclusively available until every
    /// tensor and command using it is dropped. Offset must satisfy native Vulkan
    /// storage-buffer alignment. Access/stage masks must describe its last use.
    /// External queues must be synchronized before and after Rust use. Neither
    /// the buffer nor its memory is freed by this API. Device adoption is absent.
    pub unsafe fn import_buffer(
        &self,
        buffer: u64,
        offset: usize,
        capacity: usize,
        elements: usize,
        access: u32,
        stage: u32,
    ) -> Result<Tensor> {
        if buffer == 0
            || elements == 0
            || elements > i32::MAX as usize
            || elements.checked_mul(4).is_none_or(|n| n > capacity)
        {
            return Err(Error::invalid("Invalid external buffer range"));
        }
        self.tensor(
            unsafe {
                sys::ncnn_rs_gpu_import(
                    self.0.raw,
                    buffer,
                    offset,
                    capacity,
                    elements as i32,
                    access,
                    stage,
                )
            },
            "Vulkan import",
        )
    }
}
impl Tensor {
    /// Borrows completed resident storage without mapping it.
    pub fn guard(&self) -> TensorGuard<'_> {
        TensorGuard { tensor: self }
    }
    /// Returns logical native storage bytes including native tensor padding.
    ///
    /// This is the amount the allocator accounts for, so it can exceed the element count times the
    /// element width. [`Allocator::tensor_bytes`] sums the same figure.
    pub fn storage_bytes(&self) -> usize {
        unsafe { sys::ncnn_rs_gpu_tensor_bytes(self.raw) }
    }
    /// Downloads and synchronizes into detached CPU-owned FP32 storage.
    ///
    /// The call blocks and copies, which is the cost the performance page records as the
    /// upload-download round trip. Read residency with [`TensorGuard::buffer_descriptor`] when the
    /// values must stay on the device; a refusal is reported as
    /// [`Error::NativeAt`] naming the operation.
    pub fn download(&self) -> Result<Mat> {
        let mut raw = std::ptr::null_mut();
        status(unsafe { sys::ncnn_rs_gpu_download(self.allocator.0.raw, self.raw, &mut raw) })?;
        unsafe { Mat::from_raw_owned(raw, true) }
    }
}
/// A recording owner. Outputs remain inaccessible until blocking completion.
/// Dropping or forgetting this owner cannot expose unfinished output storage.
/// The pinned native API does not provide nonblocking submit or fence polling.
///
/// Recording retains a network borrow until completion or cancellation.
///
/// ```compile_fail
/// use ncnn_rust::{gpu::Allocator, Net};
/// fn cannot_clear(allocator: &Allocator, net: &mut Net) {
///     let command = allocator.command(net).unwrap();
///     net.clear();
///     command.wait().unwrap();
/// }
/// ```
pub struct Command<'a> {
    raw: *mut c_void,
    allocator: Allocator,
    _net: &'a Net,
    inputs: Vec<&'a Tensor>,
    outputs: Vec<Tensor>,
    recorded: bool,
    failed: bool,
}
impl<'a> Command<'a> {
    /// Records one resident input for this command.
    ///
    /// The tensor must belong to the same device as the allocator, and inputs have to be recorded
    /// before the first [`extract`](Self::extract). A rejected recording leaves the command in a
    /// failed state that only dropping it resolves.
    pub fn input(&mut self, name: &str, tensor: &'a Tensor) -> Result<()> {
        if self.failed {
            return Err(Error::invalid("Recording failed and must be cancelled"));
        }
        if tensor.allocator.0.device.index != self.allocator.0.device.index {
            return Err(Error::invalid("Tensor device differs"));
        }
        if self.recorded {
            return Err(Error::invalid(
                "Inputs cannot change after recording outputs",
            ));
        }
        let name = CString::new(name)?;
        self.inputs.push(tensor);
        let result =
            status(unsafe { sys::ncnn_rs_gpu_command_input(self.raw, name.as_ptr(), tensor.raw) });
        self.failed = result.is_err();
        result
    }
    /// Records an output and returns its index in the completed output vector.
    ///
    /// The output storage belongs to the allocator from this call on, so it counts towards
    /// [`Allocator::tensor_bytes`] before any work is submitted. Recording an output closes input
    /// recording for this command.
    pub fn extract(&mut self, name: &str) -> Result<usize> {
        if self.failed {
            return Err(Error::invalid("Recording failed and must be cancelled"));
        }
        let name = CString::new(name)?;
        self.recorded = true;
        let raw = unsafe { sys::ncnn_rs_gpu_command_extract(self.raw, name.as_ptr()) };
        match self.allocator.tensor(raw, "Vulkan command output") {
            Ok(tensor) => self.outputs.push(tensor),
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        }
        Ok(self.outputs.len() - 1)
    }
    /// Submits recorded work and blocks for native fence completion.
    ///
    /// The returned tensors are completed and safe to read through
    /// [`Tensor::download`] or [`Tensor::guard`]. A native recording failure cancels the command
    /// instead of submitting it, so a failed record never reaches the device.
    pub fn wait(mut self) -> Result<Vec<Tensor>> {
        if self.failed {
            return Err(Error::invalid("Recording failed and must be cancelled"));
        }
        status(unsafe { sys::ncnn_rs_gpu_command_wait(self.raw) })?;
        Ok(std::mem::take(&mut self.outputs))
    }
}
impl Drop for Command<'_> {
    fn drop(&mut self) {
        unsafe { sys::ncnn_rs_gpu_command_destroy(self.raw) }
    }
}

/// A borrowed view of completed resident storage. Forgetting this guard does not
/// release its tensor or grant access to pending commands. The shared borrow
/// prevents safe destruction while the guard remains usable.
pub struct TensorGuard<'a> {
    tensor: &'a Tensor,
}

/// A nonowning Vulkan buffer descriptor. Values remain valid only while the
/// tensor guard and its device remain alive.
#[derive(Debug, Clone, Copy)]
pub struct BufferDescriptor {
    /// Native `VkBuffer` handle.
    pub buffer: u64,
    /// Byte offset of the tensor inside the buffer.
    pub offset: usize,
    /// Buffer capacity in bytes.
    pub capacity: usize,
    /// Native access flags recorded for the descriptor.
    pub access: u32,
    /// Native pipeline-stage flags recorded for the descriptor.
    pub stage: u32,
}
impl TensorGuard<'_> {
    /// Synchronizes and returns an independent CPU tensor.
    pub fn download(&self) -> Result<Mat> {
        self.tensor.download()
    }
    /// Exposes a descriptor for explicitly synchronized external interoperability.
    ///
    /// # Safety
    /// External use must not mutate storage while any Rust command or tensor
    /// uses it. The caller synchronizes queues and restores the recorded access
    /// state before subsequent Rust use, and never destroys this borrowed buffer.
    pub unsafe fn buffer_descriptor(&self) -> Result<BufferDescriptor> {
        let mut d = BufferDescriptor {
            buffer: 0,
            offset: 0,
            capacity: 0,
            access: 0,
            stage: 0,
        };
        status(unsafe {
            sys::ncnn_rs_gpu_buffer_info(
                self.tensor.raw,
                &mut d.buffer,
                &mut d.offset,
                &mut d.capacity,
                &mut d.access,
                &mut d.stage,
            )
        })?;
        Ok(d)
    }
}
