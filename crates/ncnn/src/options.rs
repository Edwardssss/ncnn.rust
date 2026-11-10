use crate::{error::status, sys, Error, Result};
use std::{cell::Cell, marker::PhantomData, rc::Rc};

/// Identifies one native integer option.
///
/// Each variant maps to one field of the native option block;
/// [`Options::set`](crate::Options::set) documents the accepted range per variant.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
#[non_exhaustive]
pub enum CpuOption {
    /// Native worker count.
    Threads = 0,
    /// Releases cached intermediate blobs after use.
    LightMode,
    /// Enables Winograd convolution.
    Winograd,
    /// Enables single-precision matrix multiplication.
    Sgemm,
    /// Enables INT8 inference paths.
    Int8Inference,
    /// Enables packed element layouts.
    Packing,
    /// Enables packed FP16 storage.
    Fp16Packed,
    /// Stores activations as FP16.
    Fp16Storage,
    /// Uses FP16 arithmetic where available.
    Fp16Arithmetic,
    /// Stores activations as BF16.
    Bf16Storage,
    /// Uses the per-network blob pool.
    LocalPool,
    /// Sets the OpenMP spin block time.
    OpenmpBlocktime,
    /// Flushes denormal results to zero.
    FlushDenormals,
    /// Enables 3x3 Winograd kernels.
    Winograd23,
    /// Enables 5x5 Winograd kernels.
    Winograd43,
    /// Enables 7x7 Winograd kernels.
    Winograd63,
    /// Maps model files instead of copying them.
    MappedModelLoading,
    /// Hints the sequence length of an LLM KV cache.
    KvCacheSequenceHint,
    /// Allows Vulkan subgroup operations.
    SubgroupOps,
    /// Enables packed INT8 storage.
    Int8Packed,
    /// Stores activations as INT8.
    Int8Storage,
    /// Uses INT8 arithmetic where available.
    Int8Arithmetic,
    /// Enables packed BF16 storage.
    Bf16Packed,
    /// Uses tensor storage for Vulkan buffers.
    TensorStorage,
    /// Keeps Vulkan weights in host memory.
    WeightsInHostMemory,
    /// Prefers shader local memory.
    ShaderLocalMemory,
    /// Allows cooperative matrix instructions.
    CooperativeMatrix,
    /// Selects kernels tuned for Cortex-A53 and A55.
    A53A55OptimizedKernel,
    /// Uses FP16 uniform buffers.
    Fp16Uniform,
    /// Uses INT8 uniform buffers.
    Int8Uniform,
    /// Enables packed INT16 storage.
    Int16Packed,
    /// Stores activations as INT16.
    Int16Storage,
}

/// Independent native boolean preferences. Native capability dispatch remains authoritative.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum OptionFlag {
    /// Releases cached intermediate blobs after use.
    LightMode,
    /// Enables Winograd convolution.
    Winograd,
    /// Enables single-precision matrix multiplication.
    Sgemm,
    /// Enables INT8 inference paths.
    Int8Inference,
    /// Enables packed element layouts.
    Packing,
    /// Enables packed FP16 storage.
    Fp16Packed,
    /// Stores activations as FP16.
    Fp16Storage,
    /// Uses FP16 arithmetic where available.
    Fp16Arithmetic,
    /// Stores activations as BF16.
    Bf16Storage,
    /// Uses the per-network blob pool.
    LocalPool,
    /// Enables 3x3 Winograd kernels.
    Winograd23,
    /// Enables 5x5 Winograd kernels.
    Winograd43,
    /// Enables 7x7 Winograd kernels.
    Winograd63,
    /// Maps model files instead of copying them.
    MappedModelLoading,
    /// Allows Vulkan subgroup operations.
    SubgroupOps,
    /// Enables packed INT8 storage.
    Int8Packed,
    /// Stores activations as INT8.
    Int8Storage,
    /// Uses INT8 arithmetic where available.
    Int8Arithmetic,
    /// Enables packed BF16 storage.
    Bf16Packed,
    /// Uses tensor storage for Vulkan buffers.
    TensorStorage,
    /// Keeps Vulkan weights in host memory.
    WeightsInHostMemory,
    /// Prefers shader local memory.
    ShaderLocalMemory,
    /// Allows cooperative matrix instructions.
    CooperativeMatrix,
    /// Selects kernels tuned for Cortex-A53 and A55.
    A53A55OptimizedKernel,
    /// Uses FP16 uniform buffers.
    Fp16Uniform,
    /// Uses INT8 uniform buffers.
    Int8Uniform,
    /// Enables packed INT16 storage.
    Int16Packed,
    /// Stores activations as INT16.
    Int16Storage,
}
impl OptionFlag {
    fn key(self) -> CpuOption {
        match self {
            Self::LightMode => CpuOption::LightMode,
            Self::Winograd => CpuOption::Winograd,
            Self::Sgemm => CpuOption::Sgemm,
            Self::Int8Inference => CpuOption::Int8Inference,
            Self::Packing => CpuOption::Packing,
            Self::Fp16Packed => CpuOption::Fp16Packed,
            Self::Fp16Storage => CpuOption::Fp16Storage,
            Self::Fp16Arithmetic => CpuOption::Fp16Arithmetic,
            Self::Bf16Storage => CpuOption::Bf16Storage,
            Self::LocalPool => CpuOption::LocalPool,
            Self::Winograd23 => CpuOption::Winograd23,
            Self::Winograd43 => CpuOption::Winograd43,
            Self::Winograd63 => CpuOption::Winograd63,
            Self::MappedModelLoading => CpuOption::MappedModelLoading,
            Self::SubgroupOps => CpuOption::SubgroupOps,
            Self::Int8Packed => CpuOption::Int8Packed,
            Self::Int8Storage => CpuOption::Int8Storage,
            Self::Int8Arithmetic => CpuOption::Int8Arithmetic,
            Self::Bf16Packed => CpuOption::Bf16Packed,
            Self::TensorStorage => CpuOption::TensorStorage,
            Self::WeightsInHostMemory => CpuOption::WeightsInHostMemory,
            Self::ShaderLocalMemory => CpuOption::ShaderLocalMemory,
            Self::CooperativeMatrix => CpuOption::CooperativeMatrix,
            Self::A53A55OptimizedKernel => CpuOption::A53A55OptimizedKernel,
            Self::Fp16Uniform => CpuOption::Fp16Uniform,
            Self::Int8Uniform => CpuOption::Int8Uniform,
            Self::Int16Packed => CpuOption::Int16Packed,
            Self::Int16Storage => CpuOption::Int16Storage,
        }
    }
}

/// Native CPU option storage. `Net::configure` applies it to an unloaded network.
///
/// # Choosing between `Options` and `NetBuilder`
///
/// Both reach the same native option block. Use [`Net::builder`](crate::Net::builder) with
/// `threads`, `flag` and `option` while a network is being created: those calls consume the
/// builder and report a configuration failure before any model is read. Use [`Options`] when an
/// option set is shared or prepared elsewhere, then hand it to
/// [`Net::configure`](crate::Net::configure) once, before loading parameters.
pub struct Options {
    pub(crate) raw: sys::ncnn_option_t,
    pub(crate) low_copy: Cell<bool>,
    _thread: PhantomData<Rc<()>>,
}
impl Options {
    /// Applies a convenience preset, not a hardware capability restriction.
    ///
    /// FP16 toggles packed, storage, and arithmetic together; FP32 clears all
    /// three. Both presets clear BF16 storage. Later `set_flag` overrides win,
    /// so mixed precision preferences remain independently configurable. Native
    /// ncnn selects supported kernels and performs runtime capability dispatch.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_precision(&mut self, precision: crate::gpu::Precision) -> Result<&mut Self> {
        let enabled = i32::from(matches!(precision, crate::gpu::Precision::Fp16));
        self.set(CpuOption::Fp16Packed, enabled)?;
        self.set(CpuOption::Fp16Storage, enabled)?;
        self.set(CpuOption::Fp16Arithmetic, enabled)?;
        self.set(CpuOption::Bf16Storage, 0)
    }
    /// Retains the backward-compatible CPU, non-lightmode convenience preset.
    pub fn new() -> Result<Self> {
        let mut options = Self::native_defaults()?;
        unsafe { sys::ncnn_option_set_use_vulkan_compute(options.raw, 0) };
        options.set_flag(OptionFlag::LightMode, false)?;
        Ok(options)
    }
    /// Allocates unchanged native defaults. Safe networks select CPU unless a
    /// tracked Vulkan device is explicitly attached, independently of these flags.
    pub fn native_defaults() -> Result<Self> {
        let raw = unsafe { sys::ncnn_option_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            low_copy: Cell::new(true),
            _thread: PhantomData,
        })
    }
    /// Overrides one independent native boolean without changing other preferences.
    ///
    /// The flag maps to the same native field as the matching
    /// [`CpuOption`](crate::CpuOption), so mixing the two forms for one field is allowed and the
    /// later write wins.
    pub fn set_flag(&mut self, flag: OptionFlag, enabled: bool) -> Result<&mut Self> {
        self.set(flag.key(), i32::from(enabled))
    }
    /// Reads one boolean option.
    pub fn flag(&self, flag: OptionFlag) -> Result<bool> {
        self.get(flag.key()).map(|value| value != 0)
    }
    /// Writes one integer option after validating its range.
    ///
    /// [`CpuOption::Threads`](crate::CpuOption::Threads) must be positive,
    /// [`CpuOption::FlushDenormals`](crate::CpuOption::FlushDenormals) stays within `0..=3`,
    /// [`CpuOption::OpenmpBlocktime`](crate::CpuOption::OpenmpBlocktime) and
    /// [`CpuOption::KvCacheSequenceHint`](crate::CpuOption::KvCacheSequenceHint) accept zero, and
    /// the remaining keys accept `0` or `1`. The value reaches the native option block only after
    /// those checks, so a rejected write leaves the block unchanged.
    pub fn set(&mut self, key: CpuOption, value: i32) -> Result<&mut Self> {
        match key {
            CpuOption::Threads if value < 1 => {
                return Err(Error::invalid("Thread count must be positive"))
            }
            CpuOption::OpenmpBlocktime | CpuOption::KvCacheSequenceHint if value < 0 => {
                return Err(Error::invalid("Value must be nonnegative"))
            }
            CpuOption::FlushDenormals if !(0..=3).contains(&value) => {
                return Err(Error::invalid("Denormal mode must be 0..=3"))
            }
            CpuOption::Threads
            | CpuOption::OpenmpBlocktime
            | CpuOption::KvCacheSequenceHint
            | CpuOption::FlushDenormals => {}
            _ if value != 0 && value != 1 => {
                return Err(Error::invalid("Boolean option must be 0 or 1"))
            }
            _ => {}
        }
        unsafe {
            status(sys::ncnn_rs_option_set(self.raw, key as i32, value))?;
        }
        Ok(self)
    }
    /// Reads one integer option.
    ///
    /// Reads come from the native block, so a value written through a raw handle is visible here.
    pub fn get(&self, key: CpuOption) -> Result<i32> {
        let mut value = 0;
        unsafe {
            status(sys::ncnn_rs_option_get(self.raw, key as i32, &mut value))?;
        }
        Ok(value)
    }
    /// Exposes advanced configuration and disables low-copy output eligibility
    /// for networks subsequently configured with these options.
    ///
    /// # Safety
    /// The caller keeps these options alive while any network uses them and
    /// restores fields the adapter does not model.
    pub unsafe fn as_raw(&self) -> sys::ncnn_option_t {
        self.low_copy.set(false);
        self.raw
    }
}
impl Drop for Options {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_option_destroy(self.raw);
        }
    }
}
