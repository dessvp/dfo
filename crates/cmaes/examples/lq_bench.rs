//! lq-CMA-ES (IPOP) evaluations to f - fopt < 1e-8 on rotated test
//! functions, median of 5 seeds, with wall time.
//! Usage: cargo run --release --example lq_bench -- sphere|elli|rosen N
use dfo_cmaes::cmaes::Rng;
use dfo_cmaes::lq::{Archive, ipop_lq};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (name, n): (&str, usize) = (&a[1], a[2].parse().unwrap());
    let mut rng = Rng::new(99);
    let mut r = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            r[i * n + j] = rng.normal();
        }
        for k in 0..i {
            let d: f64 = (0..n).map(|j| r[i * n + j] * r[k * n + j]).sum();
            for j in 0..n {
                r[i * n + j] -= d * r[k * n + j];
            }
        }
        let nn = (0..n).map(|j| r[i * n + j].powi(2)).sum::<f64>().sqrt();
        for j in 0..n {
            r[i * n + j] /= nn;
        }
    }
    let mut runs = vec![];
    let t = std::time::Instant::now();
    for seed in 0..5 {
        let best = std::cell::Cell::new(f64::INFINITY);
        let mut f = |x: &[f64]| {
            let z: Vec<f64> = (0..n).map(|i| (0..n).map(|j| r[i * n + j] * (x[j] - 0.5)).sum()).collect();
            let v = match name {
                "sphere" => z.iter().map(|t| t * t).sum::<f64>(),
                "elli" => (0..n).map(|i| 1e6f64.powf(i as f64 / (n - 1) as f64) * z[i] * z[i]).sum(),
                "rosen" => (0..n - 1).map(|i| 100.0 * (z[i] * z[i] - z[i + 1]).powi(2) + (z[i] - 1.0).powi(2)).sum(),
                _ => panic!("unknown function"),
            };
            best.set(best.get().min(v));
            v
        };
        let mut ar = Archive::default();
        let used = ipop_lq(&mut f, || best.get() < 1e-8, &vec![-5.0; n], &vec![5.0; n], 1000 * n, 100 + seed, &mut ar, None);
        runs.push(if best.get() < 1e-8 { used as f64 } else { f64::INFINITY });
    }
    runs.sort_by(f64::total_cmp);
    println!("{name} n={n}: median {} evals {:?}  ({:.2}s)", runs[2], runs, t.elapsed().as_secs_f64());
}
