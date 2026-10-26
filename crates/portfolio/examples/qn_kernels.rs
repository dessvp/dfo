//! Times the BFGS kernels: portable SIMD loops against Accelerate BLAS.
fn main() {
    println!("   n | symv portable | update portable | symv BLAS | update BLAS   (ns per call)");
    for n in [5, 10, 20, 40, 80, 160] {
        let t = dfo_portfolio::qn::bench_kernels(n, 200_000);
        println!("{n:4} | {:13.1} | {:15.1} | {:9.1} | {:11.1}", t[0], t[1], t[2], t[3]);
    }
}
