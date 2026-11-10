//! Native CPU capabilities, legacy worker masks, and OS processor-group affinity.
use crate::{error::status, sys, Error, Result};
use std::{ffi::c_void, marker::PhantomData, rc::Rc};

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
#[non_exhaustive]
/// Identifies one native CPU capability query.
pub enum CpuQuery {
    /// Counts logical processors.
    LogicalCount,
    /// Counts processors in the little cluster.
    LittleCount,
    /// Counts processors in the big cluster.
    BigCount,
    /// Counts physical cores.
    PhysicalCount,
    /// Counts physical little-cluster cores.
    PhysicalLittleCount,
    /// Counts physical big-cluster cores.
    PhysicalBigCount,
    /// Reports the L2 cache size in bytes.
    L2Bytes,
    /// Reports the L3 cache size in bytes.
    L3Bytes,
    /// Reports AVX support.
    Avx,
    /// Reports FMA support.
    Fma,
    /// Reports FMA4 support.
    Fma4,
    /// Reports XOP support.
    Xop,
    /// Reports F16C support.
    F16c,
    /// Reports AVX2 support.
    Avx2,
    /// Reports AVX-VNNI support.
    AvxVnni,
    /// Reports AVX-VNNI-INT8 support.
    AvxVnniInt8,
    /// Reports AVX-VNNI-INT16 support.
    AvxVnniInt16,
    /// Reports AVX-NE-CONVERT support.
    AvxNeConvert,
    /// Reports AVX-512 support.
    Avx512,
    /// Reports AVX-512 VNNI support.
    Avx512Vnni,
    /// Reports AVX-512 BF16 support.
    Avx512Bf16,
    /// Reports AVX-512 FP16 support.
    Avx512Fp16,
    /// Reports ARM EDSP support.
    ArmEdsp,
    /// Reports ARM NEON support.
    ArmNeon,
    /// Reports ARM VFPv4 support.
    ArmVfpv4,
    /// Reports ARM FP16 support.
    ArmFp16,
    /// Reports whether ARM CPUID queries are available.
    ArmCpuid,
    /// Reports ARM dot-product support.
    ArmDotProduct,
    /// Reports ARM FP16 multiply-accumulate support.
    ArmFp16Fma,
    /// Reports ARM BF16 support.
    ArmBf16,
    /// Reports ARM INT8 matrix-multiply support.
    ArmI8mm,
    /// Reports ARM SVE support.
    ArmSve,
    /// Reports ARM SVE2 support.
    ArmSve2,
    /// Reports ARM SVE BF16 support.
    ArmSveBf16,
    /// Reports ARM SVE INT8 matrix-multiply support.
    ArmSveI8mm,
    /// Reports ARM SVE FP32 matrix-multiply support.
    ArmSveF32mm,
}

/// Returns the native answer for one query, zero when the feature is unsupported.
///
/// The value is a native capability report rather than a Rust-side guess: counts and cache sizes
/// come back as numbers, and a feature test returns `1` when the current CPU has it and `0`
/// otherwise. Nothing here gates behavior; ncnn performs its own dispatch at run time.
pub fn query(key: CpuQuery) -> i32 {
    unsafe { sys::ncnn_rs_cpu_query(key as i32) }
}

/// An OS processor identity. Linux uses group zero and the kernel CPU index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessorId {
    /// Windows processor group; Linux reports zero.
    pub group: u32,
    /// Processor index inside the group.
    pub index: u32,
}

/// Reports support for exact OS processor identities and affinity masks.
pub fn supports_thread_affinity() -> bool {
    unsafe { sys::ncnn_rs_os_affinity_supported() != 0 }
}

fn processors(current_thread: bool) -> Result<Vec<ProcessorId>> {
    if !supports_thread_affinity() {
        return Err(Error::Unsupported(
            "exact processor identities and affinity",
        ));
    }
    for _ in 0..3 {
        let count =
            unsafe { sys::ncnn_rs_os_processors(current_thread as i32, std::ptr::null_mut(), 0) };
        if count == usize::MAX {
            return Err(Error::Native(-1));
        }
        let mut ids = vec![0u64; count];
        let actual = unsafe {
            sys::ncnn_rs_os_processors(current_thread as i32, ids.as_mut_ptr(), ids.len())
        };
        if actual == usize::MAX {
            return Err(Error::Native(-1));
        }
        if actual > ids.len() {
            continue;
        }
        ids.truncate(actual);
        return Ok(ids
            .into_iter()
            .map(|id| ProcessorId {
                group: (id >> 32) as u32,
                index: id as u32,
            })
            .collect());
    }
    Err(Error::invalid("CPU topology changed during enumeration"))
}

/// Enumerates online processors across Windows groups or Linux kernel CPU IDs.
pub fn online_processors() -> Result<Vec<ProcessorId>> {
    processors(false)
}

/// Reads the calling thread's OS mask; Windows reports its current processor group.
pub fn current_thread_affinity() -> Result<Vec<ProcessorId>> {
    processors(true)
}

/// Counts physical cores from OS topology records independently of ncnn's legacy masks.
pub fn physical_core_count() -> Result<usize> {
    let count = unsafe { sys::ncnn_rs_os_physical_count() };
    if count < 0 {
        return Err(Error::Native(count));
    }
    Ok(count as usize)
}

/// Sets and verifies the calling thread's affinity. Windows accepts one group per call.
/// Native OpenMP worker assignment remains a separate runtime operation.
pub fn set_current_thread_affinity(processors: &[ProcessorId]) -> Result<()> {
    if !supports_thread_affinity() {
        return Err(Error::Unsupported("exact current-thread affinity"));
    }
    let ids: Vec<_> = processors
        .iter()
        .map(|id| ((id.group as u64) << 32) | id.index as u64)
        .collect();
    unsafe { status(sys::ncnn_rs_thread_affinity_set(ids.as_ptr(), ids.len())) }
}

/// An owned native CPU mask. Applying it is an explicit runtime operation.
pub struct CpuSet {
    raw: *mut c_void,
    _thread: PhantomData<Rc<()>>,
}
impl CpuSet {
    /// Returns the native mask capacity in CPUs.
    pub fn capacity() -> usize {
        unsafe { sys::ncnn_rs_cpuset_capacity() as usize }
    }
    /// Builds the native power-save mask for the given mode.
    pub fn for_powersave(mode: u8) -> Result<Self> {
        let mask = Self::new()?;
        unsafe {
            status(sys::ncnn_rs_cpuset_affinity(mask.raw, mode as i32))?;
        }
        Ok(mask)
    }
    /// Creates an empty mask.
    pub fn new() -> Result<Self> {
        let raw = unsafe { sys::ncnn_rs_cpuset_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }
    /// Enables or disables one CPU index.
    pub fn set(&mut self, cpu: usize, enabled: bool) -> Result<()> {
        let cpu =
            i32::try_from(cpu).map_err(|_| Error::invalid("CPU index exceeds native range"))?;
        unsafe { status(sys::ncnn_rs_cpuset_update(self.raw, cpu, enabled as i32)) }
    }
    /// Reports whether one CPU index is enabled.
    pub fn contains(&self, cpu: usize) -> Result<bool> {
        let cpu =
            i32::try_from(cpu).map_err(|_| Error::invalid("CPU index exceeds native range"))?;
        let result = unsafe { sys::ncnn_rs_cpuset_contains(self.raw, cpu) };
        if result < 0 {
            return Err(Error::invalid("CPU index exceeds native mask capacity"));
        }
        Ok(result != 0)
    }
    /// Counts enabled CPU indices.
    pub fn count(&self) -> usize {
        unsafe { sys::ncnn_rs_cpuset_count(self.raw) as usize }
    }
    /// Applies the mask to native workers.
    ///
    /// # Safety
    /// Serialize changes with native inference and other users of global thread settings.
    pub unsafe fn apply(&self) -> Result<()> {
        if !supports_thread_affinity() {
            return Err(Error::Unsupported("exact native worker affinity"));
        }
        status(sys::ncnn_rs_cpuset_apply(self.raw))
    }
}
impl Drop for CpuSet {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_rs_cpuset_destroy(self.raw);
        }
    }
}
