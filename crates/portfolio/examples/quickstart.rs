//! The README's usage example: minimize a shifted 10-D sphere on [-5, 5]^10 to 1e-10.
fn main() {
    let n = 10;
    let (lo, hi) = (vec![-5.0; n], vec![5.0; n]);
    let best = std::cell::Cell::new(f64::INFINITY);
    let sphere = |x: &[f64]| {
        let v = x.iter().map(|t| (t - 1.234).powi(2)).sum::<f64>();
        best.set(best.get().min(v));
        v
    };
    let cfg = dfo_portfolio::HybridConfig::default();
    let used = dfo_portfolio::minimize(sphere, || best.get() < 1e-10, &lo, &hi, f64::NEG_INFINITY, 100_000, &cfg);
    println!("f = {:.3e} after {used} evaluations", best.get());
}
