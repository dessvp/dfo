//! Evaluations the quasi-Newton phase needs on an f10-like problem: a
//! rotated ellipsoid (condition 1e6) with the BBOB oscillation transform.
//! Usage: cargo run --release --example qn_probe -- [n] [budget]

use dfo_cmaes::cmaes::Rng;
use dfo_cmaes::lq::Archive;
use dfo_portfolio::qn;

fn tosz(x: f64) -> f64 {
    if std::env::var_os("NOTOSZ").is_some() {
        return x;
    }
    if x == 0.0 {
        return 0.0;
    }
    let h = x.abs().ln();
    let (c1, c2) = if x > 0.0 { (10.0, 7.9) } else { (5.5, 3.1) };
    x.signum() * (h + 0.049 * ((c1 * h).sin() + (c2 * h).sin())).exp()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).map_or(20, |s| s.parse().unwrap());
    let budget: usize = a.get(2).map_or(2000, |s| s.parse().unwrap());
    let mut rng = Rng::new(7);
    // Random orthogonal matrix by Gram-Schmidt.
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
    let xopt: Vec<f64> = (0..n).map(|_| 8.0 * rng.uniform() - 4.0).collect();
    let fopt = 79.48;
    let trace = std::cell::RefCell::new(Vec::new());
    let rosen = std::env::var_os("ROSEN").is_some();
    let c = ((n as f64).sqrt() / 8.0).max(1.0);
    let mut f = |x: &[f64]| {
        let fx = if rosen {
            // BBOB f9: Rosenbrock of z = c R x + 1/2.
            let z: Vec<f64> = (0..n).map(|i| c * (0..n).map(|j| r[i * n + j] * x[j]).sum::<f64>() + 0.5).collect();
            fopt + (0..n - 1).map(|i| 100.0 * (z[i] * z[i] - z[i + 1]).powi(2) + (z[i] - 1.0).powi(2)).sum::<f64>()
        } else {
            fopt + (0..n)
                .map(|i| {
                    let z = tosz((0..n).map(|j| r[i * n + j] * (x[j] - xopt[j])).sum());
                    1e6f64.powf(i as f64 / (n - 1) as f64) * z * z
                })
                .sum::<f64>()
        };
        let mut t = trace.borrow_mut();
        let best = t.last().map_or(f64::INFINITY, |&(_, b): &(usize, f64)| b).min(fx - fopt);
        let k = t.len() + 1;
        t.push((k, best));
        fx
    };
    let x0: Vec<f64> = (0..n).map(|_| if std::env::var_os("X0ZERO").is_some() { 0.0 } else { 8.0 * rng.uniform() - 4.0 }).collect();
    let f0 = f(&x0);
    let mut ar = Archive::default();
    let res = qn::bfgs(
        &mut f,
        &mut || false,
        &x0,
        f0,
        &vec![-std::env::var("BOX").map_or(5.0, |v| v.parse().unwrap()); n],
        &vec![std::env::var("BOX").map_or(5.0, |v| v.parse().unwrap()); n],
        budget,
        std::env::var("QNH0").map_or(1.0, |v| v.parse().unwrap()),
        &mut ar,
    );
    let t = trace.borrow();
    for target in [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8] {
        let k = t.iter().find(|&&(_, b)| b <= target).map(|&(k, _)| k);
        println!("target {target:8.0e}: {}", k.map_or("-".into(), |k| format!("{k} evals ({:.0}·D)", k as f64 / n as f64)));
    }
    println!("used {} final f-fopt {:.3e}", res.used, res.f - fopt);
}
