//! Surrogate-fit kernel timings: portable NEON vs Accelerate (AMX).
fn main() {
    for (n, p, m) in [(10, 66, 100), (20, 231, 347), (40, 861, 1292)] {
        let t = dfo_cmaes::lq::bench_kernels(p, m, 20);
        println!(
            "{n:2}-D p={p:3} m={m:4}  A'A: neon {:8.1} us  amx {:8.1} us ({:4.1}x)   cholesky: neon {:8.1} us  amx {:8.1} us ({:4.1}x)",
            t[0] * 1e6,
            t[1] * 1e6,
            t[0] / t[1],
            t[2] * 1e6,
            t[3] * 1e6,
            t[2] / t[3]
        );
    }
}
