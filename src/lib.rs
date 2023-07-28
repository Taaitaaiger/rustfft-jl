//! Glue code to call RustFFT from Julia
//!
//! "RustFFT is a high-performance, SIMD-accelerated FFT library written in pure Rust. It can
//! compute FFTs of any size, including prime-number sizes, in O(nlogn) time." - RustFFT readme
//!
//! The content of this module is exported to Julia using the [julia_module] macro from [jlrs], or
//! otherwise serves to facilitate interfacing between RustFFT and Julia. This code is not
//! distributed as a crate but as a JLL. You can read more about JLLs [here]. The build recipe can
//! be found [in the Yggdrasil repository]. Items exposed by this library can be accessed by using
//! the [RustFFT package].
//!
//! [here]: https://docs.binarybuilder.org/stable/
//! [in the Yggdrasil repository]: https://github.com/JuliaPackaging/Yggdrasil/blob/master/R/rustfft/build_tarballs.jl
//! [RustFFT package]: https://github.com/Taaitaaiger/RustFFT.jl

use std::sync::Arc;

use jlrs::{
    convert::compatible::{Compatible, CompatibleCast},
    data::{
        layout::valid_layout::ValidField,
        managed::{
            array::TypedRankedArray,
            value::typed::{TypedValue, TypedValueRet},
        },
        types::{construct_type::ConstructType, foreign_type::OpaqueType},
    },
    error::JlrsError,
    prelude::*,
};

use rustfft::{
    num_complex::{self},
    Fft, FftNum, FftPlanner as FftPlannerImp,
};

/// Dummy implementation for _Unwind_Resume
///
/// In order to work around an incompatibility with mingw, _Unwind_Resume must be implemented.
/// See: https://github.com/rust-lang/rust/issues/79609#issuecomment-815025991
///
/// This function will never be called because this library is always compiled with `panic=abort`.
#[cfg(all(target_arch = "x86", target_os = "windows", target_env = "gnu"))]
#[allow(unused)]
#[no_mangle]
unsafe extern "C" fn _Unwind_Resume() {}

/// Layout for `Base.Complex`, generated with Jlrs.Reflect:
///
/// ```ignore
/// using Jlrs.Reflect
/// layout = reflect([Complex])
/// renamestruct!(layout, Complex, "JuliaComplex")
/// layout
/// ```
#[repr(C)]
#[derive(
    Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn,
)]
#[jlrs(julia_type = "Base.Complex")]
pub struct JuliaComplex<T> {
    pub re: T,
    pub im: T,
}

// Safety: RustFFT uses `num_complex::Complex` which is compatible with `JuliaComplex`, both are
// repr(C) and have the same field layout.
unsafe impl<T> Compatible<num_complex::Complex<T>> for JuliaComplex<T> where T: ValidField + Clone {}

/// A type that wraps an `FftPlanner` from RustFFT so the `OpaqueType` trait can be implemented.
pub struct FftPlanner<T: FftNum>(FftPlannerImp<T>);

// RustFFT supports 32 and 64 bit floating point numbers, so we'll create aliases for both types
// and implement `OpaqueType` for them. Implementing this trait is safe because the planner
// contains no references to Julia data.

/// A planner for 32-bits complex numbers.
pub type FftPlanner32 = FftPlanner<f32>;
unsafe impl OpaqueType for FftPlanner32 {}

/// A planner for 64-bits complex numbers.
pub type FftPlanner64 = FftPlanner<f64>;
unsafe impl OpaqueType for FftPlanner64 {}

impl<T: FftNum> FftPlanner<T>
where
    Self: OpaqueType,
    FftInstance<T>: OpaqueType,
{
    /// Create a new instance of `FftPlanner`,
    #[inline]
    pub fn new() -> TypedValueRet<Self> {
        // Safety: this function is called through `ccall`, the leaked data is returned
        // immediately after it has been constructed so there's no reason to root it.
        unsafe {
            CCall::stackless_invoke(|unrooted| {
                let planner = FftPlanner(FftPlannerImp::new());
                TypedValue::new(unrooted, planner).leak()
            })
        }
    }

    /// Return an `FftInstance` that computes forward FFTs of size `len`.
    ///
    /// This method takes a mutable reference to `self`, it must return a `RustResult` because this
    /// mutable borrow is automatically tracked to prevent aliasing exclusive references. If the
    /// instance is already borrowed, a `BorrowError` is thrown.
    #[inline]
    pub fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::stackless_invoke(|unrooted| {
                let instance = self.0.plan_fft_forward(len);
                let instance = FftInstance { instance, len };
                TypedValue::new(unrooted, instance).leak()
            })
        }
    }

    /// Return an `FftInstance` that computes inverse FFTs of size `len`.
    ///
    /// The same further comments as those provided for `plan_fft_forward` apply.
    #[inline]
    pub fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::stackless_invoke(|unrooted| {
                let instance = self.0.plan_fft_inverse(len);
                let instance = FftInstance { instance, len };
                TypedValue::new(unrooted, instance).leak()
            })
        }
    }

    /// Return an `FftInstance` that computes either  forward or inverse FFTs of size `len`.
    ///
    /// The second argument, `direction`, must be either `:forward` or `:inverse`. Oherwise an
    /// `ErrorException` is thrown. The same further comments as those provided for
    /// `plan_fft_forward` apply.
    #[inline]
    pub fn plan_fft(
        &mut self,
        direction: Symbol,
        len: usize,
    ) -> JlrsResult<TypedValueRet<FftInstance<T>>> {
        match direction.as_str()? {
            "forward" => Ok(self.plan_fft_forward(len)),
            "inverse" => Ok(self.plan_fft_inverse(len)),
            _ => Err(JlrsError::exception(
                "direction must be :forward or :inverse",
            ))?,
        }
    }
}

/// An instance of [`Fft`]
///
/// The methods of [`FftPlanner`] return  instances of this type, which can be used to actually
/// compute the FFT in the given direction. Like `FftPlanner` a newtype and aliases are defined,
/// and the `OpaqueType` trait can be trivially implemented for both aliases.
pub struct FftInstance<T> {
    instance: Arc<dyn Fft<T>>,
    len: usize,
}

/// An alias for instances of [`FftInstance`] that support 32-bits complex numbers.
pub type FftInstance32 = FftInstance<f32>;
unsafe impl OpaqueType for FftInstance32 {}

/// An alias for instances of [`FftInstance`] that support 64-bits complex numbers.
pub type FftInstance64 = FftInstance<f64>;
unsafe impl OpaqueType for FftInstance64 {}

impl<T> FftInstance<T>
where
    T: FftNum + ValidField + Clone + ConstructType,
    Self: OpaqueType,
{
    /// Calcalate the forward or inverse FFT of `buffer`.
    ///
    /// If the data is already borrowed or the size requirements haven't been met an error is
    /// returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is tracked, but that only
    /// guarantees Rust functions are aware the data is mutably borrowed.
    #[inline]
    pub unsafe fn process(&self, buffer: TypedRankedArray<JuliaComplex<T>, 1>) -> JlrsResult<()> {
        let mut arr = buffer.as_typed();
        // Try to mutably track the buffer. Return a `BorrowError` if this fails.
        let mut tracked_buffer = arr.track_exclusive()?;

        // Access the array as a mutable slice and convert its type to the compatible
        // `num_complex::Complex`.
        let slice = tracked_buffer
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T>>();

        // https://docs.rs/rustfft/6.1.0/src/rustfft/lib.rs.html#183
        // This method panics if:
        // - `buffer.len() % self.len() > 0`
        // - `buffer.len() < self.len()`
        let len = slice.len();
        let fft_len = self.len;
        if len < fft_len || len % fft_len > 0 {
            Err(JlrsError::exception("Invalid length"))?;
        }

        // Transform the slice to its (inverse) FFT
        self.instance.process(slice);

        // Success!
        Ok(())
    }

    /// Calcalate the forward or inverse FFT of `buffer` without tracking the array.
    ///
    /// If the size requirements haven't been met an error is returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer isn't tracked, so other Rust
    /// functions are also unaware of this mutable borrow.
    #[inline]
    pub unsafe fn process_untracked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<T>, 1>,
    ) -> JlrsResult<()> {
        let mut buffer = buffer.as_typed();
        let mut accessor = buffer.bits_data_mut().unwrap_unchecked();
        let slice = accessor
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T>>();

        // https://docs.rs/rustfft/6.1.0/src/rustfft/lib.rs.html#183
        // This method panics if:
        // - `buffer.len() % self.len() > 0`
        // - `buffer.len() < self.len()`
        let len = slice.len();
        let fft_len = self.len;
        if len < fft_len || len % fft_len > 0 {
            Err(JlrsError::exception("Invalid length"))?;
        }

        // Transform the slice to its (inverse) FFT
        self.instance.process(slice);

        // Success!
        Ok(())
    }

    /// Calcalate the forward or inverse FFT of `buffer` without checking any requirements.
    ///
    /// If the size requirements aren't met, the function will panic. In practice this means the
    /// process aborts.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer isn't tracked, so other Rust
    /// functions are also unaware of this mutable borrow.
    #[inline]
    pub unsafe fn process_unchecked(&self, buffer: TypedRankedArray<JuliaComplex<T>, 1>) {
        let mut buffer = buffer.as_typed();
        let mut accessor = buffer.bits_data_mut().unwrap_unchecked();
        let slice = accessor
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T>>();
        self.instance.process(slice);
    }
}

julia_module! {
    become rustfft_jl_init;

    #[doc = "    FftPlanner32"]
    #[doc = ""]
    #[doc = "A planner for forward and inverse FFTs of `Vector{Complex{Float32}}` data. A new planner can"]
    #[doc = "be created by calling the zero-argument constructor."]
    struct FftPlanner32;
    in FftPlanner32 fn new() -> TypedValueRet<FftPlanner32> as FftPlanner32;

    #[doc= "    rustfft_plan_fft_forward(planner, len::UInt)"]
    #[doc= ""]
    #[doc = "Plan a forward FFT of length `len`. Returns either an `FftInstance32` or `FftInstance64`"]
    #[doc = "depending on the provided planner, which must be either an `FftPlanner32` or "]
    #[doc = "`FftPlanner64`."]
    in FftPlanner32 fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<f32>> as rustfft_plan_fft_forward!;

    #[doc = "    rustfft_plan_fft_inverse(planner, len::UInt)"]
    #[doc = ""]
    #[doc = "Plan an inverse FFT of length `len`. Returns either an `FftInstance32` or `FftInstance64`"]
    #[doc = "depending on the provided planner, which must be either an `FftPlanner32` or "]
    #[doc = "`FftPlanner64`."]
    in FftPlanner32 fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<f32>> as rustfft_plan_fft_inverse!;

    #[doc= "    rustfft_plan_fft(planner, direction::Symbol, len::UInt)"]
    #[doc= ""]
    #[doc = "Plan either a forward or an inverse FFT of length `len`. Returns either an `FftInstance32` or"]
    #[doc = "`FftInstance64` depending on the provided planner, which must be either an `FftPlanner32` or"]
    #[doc = "`FftPlanner64`. The direction must be either `:forward` or `:inverse`"]
    in FftPlanner32 fn plan_fft(
        &mut self,
        direction: Symbol,
        len: usize
    ) -> JlrsResult<TypedValueRet<FftInstance<f32>>> as rustfft_plan_fft!;

    #[doc = "    FftInstance32"]
    #[doc = ""]
    #[doc = "A planned FFT instance that can compute either forward or inverse FFTs of"]
    #[doc = "`Vector{Complex{Float32}}` data whose length is an integer multiple of the planned length."]
    struct FftInstance32;

    #[doc = "    rustfft_fft!(instance, data)"]
    #[doc = ""]
    #[doc = "Computes the forward or inverse FFT of the data in-place. `instance` must be either a"]
    #[doc = "`FftInstance32` or a `FftInstance64`. `data` must be either a `Vector{Complex{Float32}}` or"]
    #[doc = "a `Vector{Complex{Float64}}`, the width must match the that of the provided `instance`, its"]
    #[doc = "length must be an integer multiple the length of the `instance`."]
    #[untracked_self]
    in FftInstance32 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) -> JlrsResult<()> as rustfft_fft!;

    #[doc = "    rustfft_fft_untracked!(instance, data)"]
    #[doc = ""]
    #[doc = "Computes the forward or inverse FFT of the data in-place. `instance` must be either a"]
    #[doc = "`FftInstance32` or a `FftInstance64`. `data` must be either a `Vector{Complex{Float32}}` or"]
    #[doc = "a `Vector{Complex{Float64}}`, the width must match the that of the provided `instance`, its"]
    #[doc = "length must be an integer multiple the length of the `instance`."]
    #[untracked_self]
    in FftInstance32 fn process_untracked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) -> JlrsResult<()> as rustfft_fft_untracked!;


    #[doc = "    rustfft_fft_unchecked!(instance, data)"]
    #[doc = ""]
    #[doc = "Computes the forward or inverse FFT of the data in-place. `instance` must be either a"]
    #[doc = "`FftInstance32` or a `FftInstance64`. `data` must be either a `Vector{Complex{Float32}}` or"]
    #[doc = "a `Vector{Complex{Float64}}`, the width must match the that of the provided `instance`, its"]
    #[doc = "length must be an integer multiple the length of the `instance`."]
    #[untracked_self]
    in FftInstance32 fn process_unchecked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) as rustfft_fft_unchecked!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance32 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) -> JlrsResult<()> as rustfft_fft_gc_safe!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance32 fn process_untracked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) -> JlrsResult<()> as rustfft_fft_untracked_gc_safe!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance32 fn process_unchecked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) as rustfft_fft_unchecked_gc_safe!;

    #[doc = "    FftPlanner64"]
    #[doc = ""]
    #[doc = "A planner for forward and inverse FFTs of `Vector{Complex{Float64}}` data. A new planner can"]
    #[doc = "be created by calling the zero-argument constructor."]
    struct FftPlanner64;
    in FftPlanner64 fn new() -> TypedValueRet<FftPlanner64> as FftPlanner64;

    in FftPlanner64 fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<f64>> as rustfft_plan_fft_forward!;
    in FftPlanner64 fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<f64>> as rustfft_plan_fft_inverse!;

    in FftPlanner64 fn plan_fft(
        &mut self,
        direction: Symbol,
        len: usize
    ) -> JlrsResult<TypedValueRet<FftInstance<f64>>> as rustfft_plan_fft!;


    #[doc = "    FftInstance64"]
    #[doc = ""]
    #[doc = "A planned FFT instance that can compute either forward or inverse FFTs of "]
    #[doc = "`Vector{Complex{Float64}}` data whose length is an integer multiple of the planned length."]
    struct FftInstance64;

    #[untracked_self]
    in FftInstance64 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) -> JlrsResult<()> as rustfft_fft!;

    #[untracked_self]
    in FftInstance64 fn process_untracked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) -> JlrsResult<()> as rustfft_fft_untracked!;

    #[untracked_self]
    in FftInstance64 fn process_unchecked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) as rustfft_fft_unchecked!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance64 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) -> JlrsResult<()> as rustfft_fft_gc_safe!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance64 fn process_untracked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) -> JlrsResult<()> as rustfft_fft_untracked_gc_safe!;

    #[untracked_self]
    #[gc_safe]
    in FftInstance64 fn process_unchecked(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) as rustfft_fft_unchecked_gc_safe!;
}
