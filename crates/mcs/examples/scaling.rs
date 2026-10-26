//! Wall time per evaluation of the global phase (no local search) on
//! Hartman 6, objective included; best of several runs.
use dfo_mcs::{Config, Mcs};
use std::time::Instant;

fn hartman6(x: &[f64]) -> f64 {
    const A: [[f64; 6]; 4] =
        [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0], [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]];
    const P: [[f64; 6]; 4] = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ];
    const AL: [f64; 4] = [1.0, 1.2, 3.0, 3.2];
    -(0..4).map(|j| AL[j] * (-(0..6).map(|i| A[j][i] * (x[i] - P[j][i]).powi(2)).sum::<f64>()).exp()).sum::<f64>()
}

fn main() {
    let one: Option<usize> = std::env::args().nth(1).map(|s| s.parse().unwrap());
    let sizes: Vec<usize> = one.map_or(vec![1000, 4000, 16000, 64000, 256000], |k| vec![k]);
    for nf in sizes {
        let mut best = f64::INFINITY;
        for _ in 0..if one.is_some() { 1 } else { 7 } {
            let mut cfg = Config::new(6);
            cfg.local = false;
            let mut m = Mcs::new(&[0.0; 6], &[1.0; 6], cfg);
            let mut f = hartman6;
            let t = Instant::now();
            let st = m.run(&mut f, nf, nf);
            best = best.min(t.elapsed().as_nanos() as f64 / m.evaluations() as f64);
            if one.is_some() {
                eprintln!("{st:?} after {} evaluations, {:.2} s, stats {:?}", m.evaluations(), t.elapsed().as_secs_f64(), m.stats);
            }
        }
        println!("hm6 nf={nf:7} {best:6.0} ns/eval");
    }
}
