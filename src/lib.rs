use std::sync::Arc;

use jlrs::{
    ccall::AsyncCallback,
    convert::compatible::{Compatible, CompatibleCast},
    data::{
        layout::valid_layout::ValidField,
        managed::{
            array::{TypedRankedArray, TypedRankedArrayUnbound},
            rust_result::{RustResult, RustResultRet},
            value::typed::{TypedValue, TypedValueRet, TypedValueUnbound},
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

// Layout for `Base.Complex`, generated with Jlrs.Reflect:
//
// ```
// using Jlrs.Reflect
// layout = reflect([Complex])
// renamestruct!(layout, Complex, "JuliaComplex")
// layout
// ```
#[repr(C)]
#[derive(
    Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn,
)]
#[jlrs(julia_type = "Base.Complex")]
pub struct JuliaComplex<T> {
    pub re: T,
    pub im: T,
}

// RustFFT uses `num_complex::Complex` which is compatible with `JuliaComplex`: both are repr(C)
// and have the same field layout.
unsafe impl<T> Compatible<num_complex::Complex<T>> for JuliaComplex<T> where T: ValidField + Clone {}

// A type that wraps an `FftPlanner` from RustFFT so the `OpaqueType` trait can be implemented.
struct FftPlanner<T: FftNum>(FftPlannerImp<T>);

// RustFFT supports 32 and 64 bit floating point numbers, so we'll create aliases for both types
// and implement `OpaqueType` for them. Implementing this trait is safe because the planner
// contains no references to Julia data.
type FftPlanner32 = FftPlanner<f32>;
unsafe impl OpaqueType for FftPlanner32 {}

type FftPlanner64 = FftPlanner<f64>;
unsafe impl OpaqueType for FftPlanner64 {}

impl<T: FftNum> FftPlanner<T>
where
    Self: OpaqueType,
    FftInstance<T>: OpaqueType,
{
    // Create a new instance of `FftPlanner`,
    fn new() -> TypedValueRet<Self> {
        // Safety: this function is called through `ccall`, the leaked data is returned
        // immediately after it has been constructed so there's no reason to root it.
        unsafe {
            CCall::invoke(|frame| {
                let planner = FftPlanner(FftPlannerImp::new());
                TypedValue::new(&frame, planner).leak()
            })
        }
    }

    // Return an `FftInstance` that computes forward FFTs of size `len`.
    //
    // This method takes a mutable reference to `self`, it must return a `RustResult` because this
    // mutable borrow is automatically tracked to prevent aliasing exclusive references. If the
    // instance is already borrowed, a `BorrowError` is thrown.
    fn plan_fft_forward(&mut self, len: usize) -> RustResultRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::invoke(|mut frame| {
                let instance = TypedValue::new(
                    &mut frame,
                    FftInstance {
                        instance: self.0.plan_fft_forward(len),
                        len,
                    },
                );
                let unrooted = frame.unrooted();
                RustResult::ok(unrooted.into_extended_target(&mut frame), instance).leak()
            })
        }
    }

    // Return an `FftInstance` that computes inverse FFTs of size `len`.
    //
    // The same further comments as those provided for `plan_fft_forward` apply.
    fn plan_fft_inverse(&mut self, len: usize) -> RustResultRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::invoke(|mut frame| {
                let instance = TypedValue::new(
                    &mut frame,
                    FftInstance {
                        instance: self.0.plan_fft_inverse(len),
                        len,
                    },
                );
                let unrooted = frame.unrooted();
                RustResult::ok(unrooted.into_extended_target(&mut frame), instance).leak()
            })
        }
    }

    // Return an `FftInstance` that computes either  forward or inverse FFTs of size `len`.
    //
    // The second argument, `direction`, must be either `:forward` or `:inverse`. Oherwise an
    // `ErrorException` is thrown. The same further comments as those provided for
    // `plan_fft_forward` apply.
    fn plan_fft(&mut self, direction: Symbol, len: usize) -> RustResultRet<FftInstance<T>> {
        // Safety: this function is called through `ccall`, no other instances are created, and
        // the leaked data is returned immediately.
        unsafe {
            CCall::invoke_fallible(|mut frame| {
                let unrooted = frame.unrooted();
                match direction.as_str() {
                    Ok("forward") => {
                        let instance = TypedValue::new(
                            &mut frame,
                            FftInstance {
                                instance: self.0.plan_fft_forward(len),
                                len,
                            },
                        );
                        Ok(
                            RustResult::ok(unrooted.into_extended_target(&mut frame), instance)
                                .leak(),
                        )
                    }
                    Ok("inverse") => {
                        let instance = TypedValue::new(
                            &mut frame,
                            FftInstance {
                                instance: self.0.plan_fft_inverse(len),
                                len,
                            },
                        );
                        Ok(
                            RustResult::ok(unrooted.into_extended_target(&mut frame), instance)
                                .leak(),
                        )
                    }
                    _ => Err(JlrsError::exception(
                        "direction must be :forward or :inverse",
                    ))?,
                }
            })
        }
    }
}

// The methods of `FftPlanner` return  instances of this type, which can be used to actually
// compute the FFT in the given direction. Like `FftPlanner` a newtype and aliases are defined,
// and the `OpaqueType` trait can be trivially implemented for both aliases.
struct FftInstance<T> {
    instance: Arc<dyn Fft<T>>,
    len: usize,
}

type FftInstance32 = FftInstance<f32>;
unsafe impl OpaqueType for FftInstance32 {}

type FftInstance64 = FftInstance<f64>;
unsafe impl OpaqueType for FftInstance64 {}

impl<T> FftInstance<T>
where
    T: FftNum + ValidField + Clone + ConstructType,
    Self: OpaqueType,
{
    // Calcalate the forward or inverse FFT of `buffer`. If the data is already borrowed or the
    // size requirements haen't been met an error is returned
    fn process(&self, buffer: TypedRankedArray<JuliaComplex<T>, 1>) -> RustResultRet<Nothing> {
        unsafe {
            CCall::invoke(|mut frame| {
                let unrooted = frame.unrooted();

                // Try to mutably track the buffer. Return a `BorrowError` if this fails.
                let mut buffer = buffer.as_typed();
                let mut tracked_buffer = match buffer.track_exclusive() {
                    Ok(tracked_buffer) => tracked_buffer,
                    Err(_) => {
                        return RustResult::<Nothing>::borrow_error(
                            unrooted.into_extended_target(&mut frame),
                        )
                        .leak();
                    }
                };

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
                    let err = RustResult::<Nothing>::jlrs_error(
                        frame.as_extended_target(),
                        JlrsError::exception("Invalid length"),
                    );
                    return err.leak();
                }

                // Transform the slice to its (inverse) FFT
                self.instance.process(slice);

                // Success!
                let nothing = TypedValue::new(&mut frame, Nothing);
                RustResult::ok(unrooted.into_extended_target(&mut frame), nothing).leak()
            })
        }
    }
}

fn process_async<T: FftNum + ValidField + Clone + ConstructType>(
    instance: TypedValueUnbound<FftInstance<T>>,
    data: TypedRankedArrayUnbound<JuliaComplex<T>, 1>,
) -> JlrsResult<impl AsyncCallback<Nothing>>
where
    FftInstance<T>: OpaqueType,
{
    unsafe {
        let instance_tracked = instance
            .as_value()
            .track_exclusive_unbound::<FftInstance<T>>()?;
        let mut array_tracked = data.as_typed().track_exclusive_unbound()?;

        Ok(move || {
            let array_slice = array_tracked.as_mut_slice().compatible_cast_mut();
            instance_tracked.instance.process(array_slice);
            Ok(Nothing)
        })
    }
}

julia_module! {
    become rustfft_jl_init;

    #[doc = "   FftPlanner32"]
    #[doc = ""]
    #[doc = "A planner for forward and inverse FFTs of `Vector{Complex{Float32}}` data. A new planner can"]
    #[doc = "be created by calling the zero-argument constructor."]
    struct FftPlanner32;
    in FftPlanner32 fn new() -> TypedValueRet<FftPlanner32> as FftPlanner32;

    #[doc= "    plan_fft_forward(planner, len::UInt)"]
    #[doc= ""]
    #[doc = "Plan a forward FFT of length `len`. Returns either an `FftInstance32` or `FftInstance64`"]
    #[doc = "depending on the provided planner, which must be either an `FftPlanner32` or "]
    #[doc = "`FftPlanner64`."]
    in FftPlanner32 fn plan_fft_forward(&mut self, len: usize) -> RustResultRet<FftInstance<f32>>;

    #[doc = "    plan_fft_inverse(planner, len::UInt)"]
    #[doc = ""]
    #[doc = "Plan an inverse FFT of length `len`. Returns either an `FftInstance32` or `FftInstance64`"]
    #[doc = "depending on the provided planner, which must be either an `FftPlanner32` or "]
    #[doc = "`FftPlanner64`."]
    in FftPlanner32 fn plan_fft_inverse(&mut self, len: usize) -> RustResultRet<FftInstance<f32>>;

    #[doc= "    plan_fft(planner, direction::Symbol, len::UInt)"]
    #[doc= ""]
    #[doc = "Plan either a forward or an inverse FFT of length `len`. Returns either an `FftInstance32` or"]
    #[doc = "`FftInstance64` depending on the provided planner, which must be either an `FftPlanner32` or"]
    #[doc = "`FftPlanner64`. The direction must be either `:forward` or `:inverse`"]
    in FftPlanner32 fn plan_fft(
        &mut self,
        direction: Symbol,
        len: usize
    ) -> RustResultRet<FftInstance<f32>>;

    #[doc = "   FftInstance32"]
    #[doc = ""]
    #[doc = "A planned FFT instance that can compute either forward or inverse FFTs of"]
    #[doc = "`Vector{Complex{Float32}}` data whose length is an integer multiple of the planned length."]
    struct FftInstance32;

    #[doc = "   fft!(instance, data)"]
    #[doc = ""]
    #[doc = "Computes the forward or inverse FFT of the data in-place. `instance` must be either a"]
    #[doc = "`FftInstance32` or a `FftInstance64`. `data` must be either a `Vector{Complex{Float32}}` or"]
    #[doc = "a `Vector{Complex{Float64}}`, the width must match the that of the provided `instance`, its"]
    #[doc = "length must be an integer multiple the length of the `instance`."]
    in FftInstance32 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f32>, 1>
    ) -> RustResultRet<Nothing> as fft!;

    #[doc = "   FftPlanner64"]
    #[doc = ""]
    #[doc = "A planner for forward and inverse FFTs of `Vector{Complex{Float64}}` data. A new planner can"]
    #[doc = "be created by calling the zero-argument constructor."]
    struct FftPlanner64;
    in FftPlanner64 fn new() -> TypedValueRet<FftPlanner64> as FftPlanner64;

    in FftPlanner64 fn plan_fft_forward(&mut self, len: usize) -> RustResultRet<FftInstance<f64>>;
    in FftPlanner64 fn plan_fft_inverse(&mut self, len: usize) -> RustResultRet<FftInstance<f64>>;
    in FftPlanner64 fn plan_fft(
        &mut self,
        direction: Symbol,
        len: usize
    ) -> RustResultRet<FftInstance<f64>>;

    #[doc = "   FftInstance64"]
    #[doc = ""]
    #[doc = "A planned FFT instance that can compute either forward or inverse FFTs of "]
    #[doc = "`Vector{Complex{Float64}}` data whose length is an integer multiple of the planned length."]
    struct FftInstance64;
    in FftInstance64 fn process(
        &self,
        buffer: TypedRankedArray<JuliaComplex<f64>, 1>
    ) -> RustResultRet<Nothing> as fft!;

    #[doc = "   fft_async!"]
    #[doc = ""]
    #[doc = "Computes the forward or inverse FFT of the data in-place, the transform is computed on a background"]
    #[doc = "thread. See RustFFT.fft! for more info."]
    async fn process_async(
        instance: TypedValueUnbound<FftInstance<f32>>,
        data: TypedRankedArrayUnbound<JuliaComplex<f32>, 1>,
    ) -> JlrsResult<impl AsyncCallback<Nothing>> as fft_async!;

    async fn process_async(
        instance: TypedValueUnbound<FftInstance<f64>>,
        data: TypedRankedArrayUnbound<JuliaComplex<f64>, 1>,
    ) -> JlrsResult<impl AsyncCallback<Nothing>> as fft_async!;
}
