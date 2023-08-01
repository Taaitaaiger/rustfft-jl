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
        types::{
            construct_type::ConstructType,
            foreign_type::{ParametricBase, ParametricVariant},
        },
    },
    error::JlrsError,
    impl_type_parameters, impl_variant_parameters,
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

// RustFFT uses the floating-point number type as a generic, in Julia we want the full type. This
// trait connects the complex number type to its inner type.
pub trait IsComplex: 'static + Send + Sync {
    type Inner: FftNum;
}

impl<T: FftNum> IsComplex for JuliaComplex<T> {
    type Inner = T;
}

// Safety: `num_complex::Complex` which is compatible with `JuliaComplex`, both are repr(C) and
// have the same field layout.
unsafe impl<T> Compatible<num_complex::Complex<T>> for JuliaComplex<T> where T: ValidField + Clone {}

/// Wrapper around `FftPlanner` from RustFFT.
pub struct FftPlanner<T: IsComplex>(FftPlannerImp<T::Inner>);

unsafe impl<T: IsComplex> ParametricBase for FftPlanner<T> {
    type Key = FftPlanner<JuliaComplex<f32>>;
    impl_type_parameters!('T');
}

unsafe impl<T> ParametricVariant for FftPlanner<T>
where
    T: IsComplex + ConstructType,
{
    impl_variant_parameters!(T);
}

impl<T: IsComplex> FftPlanner<T>
where
    Self: ParametricVariant,
    FftInstance<T>: ParametricVariant,
{
    /// Creates a new instance of `FftPlanner`,
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

    /// Returns an `FftInstance` that computes forward FFTs of size `len`.
    #[inline]
    pub fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::stackless_invoke(|unrooted| {
                let instance = self.0.plan_fft_forward(len);
                let instance = FftInstance(instance);
                TypedValue::new(unrooted, instance).leak()
            })
        }
    }

    /// Returns an `FftInstance` that computes inverse FFTs of size `len`.
    #[inline]
    pub fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::stackless_invoke(|unrooted| {
                let instance = self.0.plan_fft_inverse(len);
                let instance = FftInstance(instance);
                TypedValue::new(unrooted, instance).leak()
            })
        }
    }
}

pub struct FftInstance<T: IsComplex>(Arc<dyn Fft<T::Inner>>);
unsafe impl<T: IsComplex> ParametricBase for FftInstance<T> {
    type Key = FftInstance<JuliaComplex<f32>>;
    impl_type_parameters!('T');
}

unsafe impl<T> ParametricVariant for FftInstance<T>
where
    T: IsComplex + ConstructType,
{
    impl_variant_parameters!(T);
}

impl<T> FftInstance<T>
where
    T: IsComplex + ValidField + Clone + ConstructType + Compatible<num_complex::Complex<T::Inner>>,
    Self: ParametricVariant,
{
    /// Returns the length required by this instance.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Calcalates the forward or inverse FFT of `buffer`.
    ///
    /// If the data is already borrowed or the size requirements haven't been met an error is
    /// returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is tracked, but that only
    /// guarantees Rust functions are aware the data is mutably borrowed.
    #[inline]
    pub unsafe fn process(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> {
        let mut arr = buffer.as_typed();
        // Try to mutably track the buffer. Return a `BorrowError` if this fails.
        let mut tracked_buffer = arr.track_exclusive()?;

        // Access the array as a mutable slice and convert its type to the compatible
        // `num_complex::Complex`.
        let slice = tracked_buffer
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T::Inner>>();

        // https://docs.rs/rustfft/6.1.0/src/rustfft/lib.rs.html#183
        // This method panics if:
        // - `buffer.len() % self.len() > 0`
        // - `buffer.len() < self.len()`
        let len = slice.len();
        let fft_len = self.len();
        if len < fft_len || len % fft_len > 0 {
            Err(JlrsError::exception("Invalid length"))?;
        }

        // Transform the slice to its (inverse) FFT
        self.0.process(slice);

        // Success!
        Ok(())
    }

    /// Calcalates the forward or inverse FFT of `buffer`.
    ///
    /// If the size requirements haven't been met an error is returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is not tracked.
    #[inline]
    pub unsafe fn process_untracked(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> {
        let mut buffer = buffer.as_typed();
        let mut accessor = buffer.bits_data_mut().unwrap_unchecked();
        let slice = accessor
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T::Inner>>();

        // https://docs.rs/rustfft/6.1.0/src/rustfft/lib.rs.html#183
        // This method panics if:
        // - `buffer.len() % self.len() > 0`
        // - `buffer.len() < self.len()`
        let len = slice.len();
        let fft_len = self.len();
        if len < fft_len || len % fft_len > 0 {
            Err(JlrsError::exception("Invalid length"))?;
        }

        // Transform the slice to its (inverse) FFT
        self.0.process(slice);

        // Success!
        Ok(())
    }

    /// Calcalates the forward or inverse FFT of `buffer`.
    ///
    /// The size requirements are not checked.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is not tracked.
    #[inline]
    pub unsafe fn process_unchecked(&self, buffer: TypedRankedArray<T, 1>) {
        let mut buffer = buffer.as_typed();
        let mut accessor = buffer.bits_data_mut().unwrap_unchecked();
        let slice = accessor
            .as_mut_slice()
            .compatible_cast_mut::<num_complex::Complex<T::Inner>>();
        self.0.process(slice);
    }
}

julia_module! {
    become rustfft_jl_init;

    for T in [JuliaComplex<f32>, JuliaComplex<f64>] {
        #[doc = "    FftPlanner{T}"]
        #[doc = ""]
        #[doc = "A planner for forward and inverse FFTs of `Vector{T}`, `T` must be `ComplexF32`"]
        #[doc = "or `ComplexF64`. A new planner can be created by calling the zero-argument"]
        #[doc = "constructor. The aliases `FftPlanner32` and `FftPlanner64` are also available."]
        struct FftPlanner<T>;

        #[doc = "    rustfft_plan_fft_forward!(planner::FftPlanner{T}, len::UInt)"]
        #[doc = ""]
        #[doc = "Plan a forward FFT for `Vector{T}` data of length `len`. If the planner is"]
        #[doc = "already tracked `JlrsCore.BorrowError` is thrown."]
        in FftPlanner<T> fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> as rustfft_plan_fft_forward!;

        #[doc = "    rustfft_plan_fft_forward_untracked!(planner::FftPlanner{T}, len::UInt)"]
        #[doc = ""]
        #[doc = "Plan a forward FFT for `Vector{T}` data of length `len` without tracking the planner."]
        #[untracked_self]
        in FftPlanner<T> fn plan_fft_forward(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> as rustfft_plan_fft_forward_untracked!;

        #[doc = "    rustfft_plan_fft_inverse!(planner::FftPlanner{T}, len::UInt)"]
        #[doc = ""]
        #[doc = "Plan an inverse FFT for `Vector{T}` data of length `len`. If the planner is"]
        #[doc = "already tracked `JlrsCore.BorrowError` is thrown."]
        in FftPlanner<T> fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> as rustfft_plan_fft_inverse!;

        #[doc = "    rustfft_plan_fft_inverse_untracked!(planner::FftPlanner{T}, len::UInt)"]
        #[doc = ""]
        #[doc = "Plan an inverse FFT for `Vector{T}` data of length `len` without tracking the planner."]
        #[untracked_self]
        in FftPlanner<T> fn plan_fft_inverse(&mut self, len: usize) -> TypedValueRet<FftInstance<T>> as rustfft_plan_fft_inverse_untracked!;

        #[doc = "    FftInstance{T}"]
        #[doc = ""]
        #[doc = "An instance of a plan to compute an FFT in some direction."]
        struct FftInstance<T>;

        #[doc = "    rustfft_plan_size(instance::FftInstance{T})"]
        #[doc = ""]
        #[doc = "The length of the vector that can be transformed with `plan`."]
        #[untracked_self]
        in FftInstance<T> fn len(&self) -> usize as rustfft_plan_size;

        #[doc = "    rustfft_fft!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place. If the array is already tracked"]
        #[doc = "or the length is incompatible `JlrsCore.JlrsError` is thrown."]
        #[untracked_self]
        in FftInstance<T> fn process(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> as rustfft_fft!;

        #[doc = "    rustfft_fft_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place. If `buffer` is already tracked"]
        #[doc = "or its length is incompatible `JlrsCore.JlrsError` is thrown. The GC is allowed"]
        #[doc = "to collect while this function is called."]
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> as rustfft_fft_gcsafe!;

        #[doc = "    rustfft_fft_untracked!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place. If the length of `buffer` is"]
        #[doc = "incompatible `JlrsCore.JlrsError` is thrown."]
        #[untracked_self]
        in FftInstance<T> fn process_untracked(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> as rustfft_fft_untracked!;

        #[doc = "    rustfft_fft_untracked_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place. If the length of `buffer` is"]
        #[doc = "incompatible `JlrsCore.JlrsError` is thrown. The GC is allowed to collect while"]
        #[doc = "this function is called."]
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process_untracked(&self, buffer: TypedRankedArray<T, 1>) -> JlrsResult<()> as rustfft_fft_untracked_gcsafe!;

        #[doc = "    rustfft_fft_unchecked!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place."]
        #[untracked_self]
        in FftInstance<T> fn process_unchecked(&self, buffer: TypedRankedArray<T, 1>) as rustfft_fft_unchecked!;

        #[doc = "    rustfft_fft_unchecked_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})"]
        #[doc = ""]
        #[doc = "Computes the planned FFT if `buffer` in-place. The GC is allowed to collect"]
        #[doc = "while this function is called."]
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process_unchecked(&self, buffer: TypedRankedArray<T, 1>) as rustfft_fft_unchecked_gcsafe!;
    };

    type FftPlanner32 = FftPlanner<JuliaComplex<f32>>;

    #[doc = "    FftPlanner32"]
    #[doc = ""]
    #[doc = "A planner for single-precision complex data."]
    in FftPlanner<JuliaComplex<f32>> fn new() -> TypedValueRet<FftPlanner<JuliaComplex<f32>>> as FftPlanner32;

    type FftPlanner64 = FftPlanner<JuliaComplex<f64>>;
    #[doc = "    FftPlanner64"]
    #[doc = ""]
    #[doc = "A planner for double-precision complex data."]
    in FftPlanner<JuliaComplex<f64>> fn new() -> TypedValueRet<FftPlanner<JuliaComplex<f64>>> as FftPlanner64;
}
