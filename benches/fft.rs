#[cfg(feature = "rt")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "rt")]
use jlrs::{data::{managed::array::TypedRankedArray, layout::complex::Complex}, memory::target::frame::LocalGcFrame, prelude::*};
#[cfg(feature = "rt")]
use pprof::{
    criterion::{Output, PProfProfiler},
    flamegraph::Options,
};

#[cfg(feature = "rt")]
use rustfft_jl::{FftInstance, FftPlanner};

#[cfg(feature = "rt")]
#[inline(never)]
fn bench_forward_fft64(frame: &LocalGcFrame<0>, c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward FFT64");
    for sz in [
        2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67,
        71, 73, 79, 83, 89, 97, 128, 256, 512, 1024, 2048, 4096, 8192,
    ] {
        frame
            .local_scope::<_, 3>(|mut frame| unsafe {
                let mut planner = FftPlanner::<Complex<f64>>::new().into_typed_value().root(&mut frame);
                let instance = planner
                    .track_exclusive()
                    .unwrap()
                    .plan_fft_forward(sz)
                    .into_typed_value()
                    .root(&mut frame)
                    .data_ptr()
                    .cast::<FftInstance<Complex<f64>>>()
                    .as_ref();

                let array = Value::eval_string(&mut frame, format!("rand(ComplexF64, {sz})"))
                    .unwrap()
                    .cast_unchecked::<TypedRankedArray<Complex<f64>, 1>>();

                group.throughput(Throughput::Elements(sz as u64));
                group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |b, _sz| {
                    b.iter(|| black_box(instance.process_unchecked(array)))
                });
            });
    }

    group.finish()
}

#[cfg(feature = "rt")]
#[inline(never)]
fn bench_forward_fft32(frame: &LocalGcFrame<0>, c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward FFT32");
    for sz in [
        2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67,
        71, 73, 79, 83, 89, 97, 128, 256, 512, 1024, 2048, 4096, 8192,
    ] {
        frame
            .local_scope::<_, 3>(|mut frame| unsafe {
                let mut planner = FftPlanner::<Complex<f32>>::new().into_typed_value().root(&mut frame);
                let instance = planner
                    .track_exclusive()
                    .unwrap()
                    .plan_fft_forward(sz)
                    .into_typed_value()
                    .root(&mut frame)
                    .data_ptr()
                    .cast::<FftInstance<Complex<f32>>>()
                    .as_ref();

                let array = Value::eval_string(&mut frame, format!("rand(ComplexF32, {sz})"))
                    .unwrap()
                    .cast_unchecked::<TypedRankedArray<Complex<f32>, 1>>();

                group.throughput(Throughput::Elements(sz as u64));
                group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |b, _sz| {
                    b.iter(|| black_box(instance.process_unchecked(array)))
                });
            });
    }

    group.finish()
}

#[cfg(feature = "rt")]
fn criterion_benchmark(c: &mut Criterion) {
    unsafe {
        let julia = Builder::new().start_local().unwrap();

        julia
            .local_scope::<_, 0>(|frame| {
                // Manually call the init function so all exported types are initialized.
                rustfft_jl::rustfft_jl_init(Module::main(&frame), 1);
                bench_forward_fft64(&frame, c);
                bench_forward_fft32(&frame, c);
            });
    }
}

#[cfg(feature = "rt")]
fn opts() -> Option<Options<'static>> {
    let mut opts = Options::default();
    opts.image_width = Some(1920);
    opts.min_width = 0.01;
    Some(opts)
}

#[cfg(feature = "rt")]
criterion_group! {
    name = fft;
    config = Criterion::default().with_profiler(PProfProfiler::new(1000, Output::Flamegraph(opts())));
    targets = criterion_benchmark
}

#[cfg(feature = "rt")]
criterion_main!(fft);

#[cfg(not(feature = "rt"))]
fn main() {}
