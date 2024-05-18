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
    data::{
        layout::{complex::Complex, valid_layout::ValidField},
        managed::{ccall_ref::CCallRefRet, value::typed::TypedValue},
        types::{
            construct_type::ConstructType,
            foreign_type::{ParametricBase, ParametricVariant},
        },
    },
    error::JlrsError,
    impl_type_parameters, impl_variant_parameters,
    prelude::*,
    weak_handle_unchecked,
};

use rustfft::{Fft, FftNum, FftPlanner as FftPlannerImp};

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

// RustFFT uses the floating-point number type as a generic, in Julia we want the full type. This
// trait connects the complex number type to its inner type.
pub trait IsComplex: 'static + Send + Sync {
    type Inner: FftNum;
}

impl<T: FftNum> IsComplex for Complex<T> {
    type Inner = T;
}

/// Wrapper around `FftPlanner` from RustFFT.
pub struct FftPlanner<T: IsComplex>(FftPlannerImp<T::Inner>);

unsafe impl<T: IsComplex> ParametricBase for FftPlanner<T> {
    type Key = FftPlanner<Complex<f32>>;
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
    pub fn new() -> CCallRefRet<Self> {
        // Safety: this function is called through `ccall`, the leaked data is returned
        // immediately after it has been constructed so there's no reason to root it.
        let handle = unsafe { weak_handle_unchecked!() };
        let planner = FftPlanner(FftPlannerImp::new());
        CCallRefRet::new(TypedValue::new(handle, planner).leak())
    }

    /// Returns an `FftInstance` that computes forward FFTs of size `len`.
    #[inline]
    pub fn plan_fft_forward(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        let handle = unsafe { weak_handle_unchecked!() };
        let instance = self.0.plan_fft_forward(len);
        let instance = FftInstance(instance);
        CCallRefRet::new(TypedValue::new(handle, instance).leak())
    }

    /// Returns an `FftInstance` that computes inverse FFTs of size `len`.
    #[inline]
    pub fn plan_fft_inverse(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        let handle = unsafe { weak_handle_unchecked!() };
        let instance = self.0.plan_fft_inverse(len);
        let instance = FftInstance(instance);
        CCallRefRet::new(TypedValue::new(handle, instance).leak())
    }
}

pub struct FftInstance<T: IsComplex>(Arc<dyn Fft<T::Inner>>);
unsafe impl<T: IsComplex> ParametricBase for FftInstance<T> {
    type Key = FftInstance<Complex<f32>>;
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
    T: IsComplex + Clone,
    T::Inner: jlrs::data::layout::is_bits::IsBits + ConstructType + ValidField,
    Self: ParametricVariant,
{
    /// Returns the length required by this instance.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Calculates the forward or inverse FFT of `buffer`.
    ///
    /// If the data is already borrowed or the size requirements haven't been met an error is
    /// returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is tracked, but that only
    /// guarantees Rust functions are aware the data is mutably borrowed.
    #[inline]
    pub unsafe fn process(&self, buffer: TypedVector<Complex<T::Inner>>) -> JlrsResult<()> {
        // Try to mutably track the buffer. Return a `BorrowError` if this fails.
        let mut tracked_buffer = buffer.track_exclusive()?;

        // Access the array as a mutable slice and convert its type to the compatible
        // `num_complex::Complex`.
        let mut slice = tracked_buffer.bits_data_mut();
        let slice = slice.as_mut_slice();

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

    /// Calculates the forward or inverse FFT of `buffer`.
    ///
    /// If the size requirements haven't been met an error is returned.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is not tracked.
    #[inline]
    pub unsafe fn process_untracked(
        &self,
        mut buffer: TypedVector<Complex<T::Inner>>,
    ) -> JlrsResult<()> {
        let mut accessor = buffer.bits_data_mut();
        let slice = accessor.as_mut_slice();

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

    /// Calculates the forward or inverse FFT of `buffer`.
    ///
    /// The size requirements are not checked.
    ///
    /// Safety: the contents of `buffer` are mutated, you must guarantee it isn't accessed from
    /// another thread while this function is called. The buffer is not tracked.
    #[inline]
    pub unsafe fn process_unchecked(&self, mut buffer: TypedVector<Complex<T::Inner>>) {
        let mut accessor = buffer.bits_data_mut();
        let slice = accessor.as_mut_slice();
        self.0.process(slice);
    }
}

julia_module! {
    become rustfft_jl_init;

    for T in [Complex<f32>, Complex<f64>] {
        ///     FftPlanner{T}
        ///
        /// A planner for forward and inverse FFTs of `Vector{T}`, `T` must be `ComplexF32`
        /// or `ComplexF64`. A new planner can be created by calling the zero-argument
        /// constructor. The aliases `FftPlanner32` and `FftPlanner64` are also available.
        struct FftPlanner<T>;

        ///     rustfft_plan_fft_forward!(planner::FftPlanner{T}, len::UInt)
        ///
        /// Plan a forward FFT for `Vector{T}` data of length `len`. If the planner is
        /// already tracked `JlrsCore.BorrowError` is thrown.
        in FftPlanner<T> fn plan_fft_forward(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> as rustfft_plan_fft_forward!;

        ///     rustfft_plan_fft_forward_untracked!(planner::FftPlanner{T}, len::UInt)
        ///
        /// Plan a forward FFT for `Vector{T}` data of length `len` without tracking the planner.
        #[untracked_self]
        in FftPlanner<T> fn plan_fft_forward(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> as rustfft_plan_fft_forward_untracked!;

        ///     rustfft_plan_fft_inverse!(planner::FftPlanner{T}, len::UInt)
        ///
        /// Plan an inverse FFT for `Vector{T}` data of length `len`. If the planner is
        /// already tracked `JlrsCore.BorrowError` is thrown.
        in FftPlanner<T> fn plan_fft_inverse(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> as rustfft_plan_fft_inverse!;

        ///     rustfft_plan_fft_inverse_untracked!(planner::FftPlanner{T}, len::UInt)
        ///
        /// Plan an inverse FFT for `Vector{T}` data of length `len` without tracking the planner.
        #[untracked_self]
        in FftPlanner<T> fn plan_fft_inverse(&mut self, len: usize) -> CCallRefRet<FftInstance<T>> as rustfft_plan_fft_inverse_untracked!;

        ///     FftInstance{T}
        ///
        /// An instance of a plan to compute an FFT in some direction.
        struct FftInstance<T>;

        ///     rustfft_plan_size(instance::FftInstance{T})
        ///
        /// The length of the vector that can be transformed with `plan`.
        #[untracked_self]
        in FftInstance<T> fn len(&self) -> usize as rustfft_plan_size;

        ///     rustfft_fft!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place. If the array is already tracked
        /// or the length is incompatible `JlrsCore.JlrsError` is thrown.
        #[untracked_self]
        in FftInstance<T> fn process(&self, buffer: TypedVector<T>) -> JlrsResult<()> as rustfft_fft!;

        ///     rustfft_fft_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place. If `buffer` is already tracked
        /// or its length is incompatible `JlrsCore.JlrsError` is thrown. The GC is allowed
        /// to collect while this function is called.
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process(&self, buffer: TypedVector<T>) -> JlrsResult<()> as rustfft_fft_gcsafe!;

        ///     rustfft_fft_untracked!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place. If the length of `buffer` is
        /// incompatible `JlrsCore.JlrsError` is thrown.
        #[untracked_self]
        in FftInstance<T> fn process_untracked(&self, buffer: TypedVector<T>) -> JlrsResult<()> as rustfft_fft_untracked!;

        ///     rustfft_fft_untracked_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place. If the length of `buffer` is
        /// incompatible `JlrsCore.JlrsError` is thrown. The GC is allowed to collect while
        /// this function is called.
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process_untracked(&self, buffer: TypedVector<T>) -> JlrsResult<()> as rustfft_fft_untracked_gcsafe!;

        ///     rustfft_fft_unchecked!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place.
        #[untracked_self]
        in FftInstance<T> fn process_unchecked(&self, buffer: TypedVector<T>) as rustfft_fft_unchecked!;

        ///     rustfft_fft_unchecked_gcsafe!(instance::FftInstance{T}, buffer::Vector{T})
        ///
        /// Computes the planned FFT if `buffer` in-place. The GC is allowed to collect
        /// while this function is called.
        #[untracked_self]
        #[gc_safe]
        in FftInstance<T> fn process_unchecked(&self, buffer: TypedVector<T>) as rustfft_fft_unchecked_gcsafe!;
    };

    type FftPlanner32 = FftPlanner<Complex<f32>>;

    ///     FftPlanner32
    ///
    /// A planner for single-precision complex data.
    in FftPlanner<Complex<f32>> fn new() -> CCallRefRet<FftPlanner<Complex<f32>>> as FftPlanner32;

    type FftPlanner64 = FftPlanner<Complex<f64>>;

    ///     FftPlanner64
    ///
    /// A planner for double-precision complex data.
    in FftPlanner<Complex<f64>> fn new() -> CCallRefRet<FftPlanner<Complex<f64>>> as FftPlanner64;
}
