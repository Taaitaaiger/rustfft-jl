#[cfg(feature = "rt")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "rt")]
use jlrs::{data::managed::array::TypedRankedArray, memory::target::frame::GcFrame, prelude::*};
#[cfg(feature = "rt")]
use pprof::{
    criterion::{Output, PProfProfiler},
    flamegraph::Options,
};

#[cfg(feature = "rt")]
use rustfft_jl::{FftInstance32, FftInstance64, FftPlanner32, FftPlanner64, JuliaComplex};

#[cfg(feature = "rt")]
#[inline(never)]
fn bench_forward_fft64(frame: &mut GcFrame, c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward FFT64");
    for sz in [
        2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67,
        71, 73, 79, 83, 89, 97, 128, 256, 512, 1024, 2048, 4096, 8192,
    ] {
        frame
            .scope(|mut frame| unsafe {
                let mut planner = FftPlanner64::new().root(&mut frame);
                let instance = planner
                    .track_exclusive()
                    .unwrap()
                    .plan_fft_forward(sz)
                    .root(&mut frame)
                    .data_ptr()
                    .cast::<FftInstance64>()
                    .as_ref();

                let array = Value::eval_string(&mut frame, format!("ones(ComplexF64, {sz})"))
                    .unwrap()
                    .cast_unchecked::<TypedRankedArray<JuliaComplex<f64>, 1>>();

                group.throughput(Throughput::Elements(sz as u64));
                group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |b, _sz| {
                    b.iter(|| black_box(instance.process_unchecked(array)))
                });
                Ok(())
            })
            .unwrap();
    }

    group.finish()
}

#[cfg(feature = "rt")]
#[inline(never)]
fn bench_forward_fft32(frame: &mut GcFrame, c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward FFT32");
    for sz in [
        2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67,
        71, 73, 79, 83, 89, 97, 128, 256, 512, 1024, 2048, 4096, 8192,
    ] {
        frame
            .scope(|mut frame| unsafe {
                let mut planner = FftPlanner32::new().root(&mut frame);
                let instance = planner
                    .track_exclusive()
                    .unwrap()
                    .plan_fft_forward(sz)
                    .root(&mut frame)
                    .data_ptr()
                    .cast::<FftInstance32>()
                    .as_ref();

                let array = Value::eval_string(&mut frame, format!("ones(ComplexF32, {sz})"))
                    .unwrap()
                    .cast_unchecked::<TypedRankedArray<JuliaComplex<f32>, 1>>();

                group.throughput(Throughput::Elements(sz as u64));
                group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |b, _sz| {
                    b.iter(|| black_box(instance.process_unchecked(array)))
                });
                Ok(())
            })
            .unwrap();
    }

    group.finish()
}

#[cfg(feature = "rt")]
fn criterion_benchmark(c: &mut Criterion) {
    unsafe {
        let mut frame = StackFrame::new();
        let mut julia = RuntimeBuilder::new().start().unwrap();
        let mut julia = julia.instance(&mut frame);

        julia
            .scope(|mut frame| {
                rustfft_jl::rustfft_jl_init(Module::main(&frame), 1);
                bench_forward_fft64(&mut frame, c);
                bench_forward_fft32(&mut frame, c);

                Ok(())
            })
            .unwrap();
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
