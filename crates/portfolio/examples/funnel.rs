//! Prototype: monotonic basin hopping on a Lunacek bi-Rastrigin function
//! (BBOB f24, from its published definition), counting how often the
//! global funnel is reached within a budget.
//! Usage: cargo run --release --example funnel -- [n] [budget] [r] [runs]

use dfo_cmaes::cmaes::{Cmaes, Rng};

struct Lunacek {
    n: usize,
    sign: Vec<f64>,
    q: Vec<f64>,
    r: Vec<f64>,
    lam: Vec<f64>,
}

fn orth(n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = rng.normal();
        }
        for k in 0..i {
            let d: f64 = (0..n).map(|j| m[i * n + j] * m[k * n + j]).sum();
            for j in 0..n {
                m[i * n + j] -= d * m[k * n + j];
            }
        }
        let nn = (0..n).map(|j| m[i * n + j].powi(2)).sum::<f64>().sqrt();
        for j in 0..n {
            m[i * n + j] /= nn;
        }
    }
    m
}

impl Lunacek {
    fn new(n: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let sign = (0..n).map(|_| if rng.uniform() < 0.5 { -1.0 } else { 1.0 }).collect();
        let lam = (0..n).map(|i| 100f64.powf(i as f64 / (2.0 * (n - 1) as f64))).collect();
        Lunacek { n, sign, q: orth(n, &mut rng), r: orth(n, &mut rng), lam }
    }

    fn f(&self, x: &[f64]) -> f64 {
        let n = self.n;
        let nf = n as f64;
        let (mu0, d) = (2.5, 1.0);
        let s = 1.0 - 1.0 / (2.0 * (nf + 20.0).sqrt() - 8.2);
        let mu1 = -((mu0 * mu0 - d) / s).sqrt();
        let xh: Vec<f64> = (0..n).map(|i| 2.0 * self.sign[i] * x[i]).collect();
        let a: f64 = xh.iter().map(|t| (t - mu0).powi(2)).sum();
        let b: f64 = d * nf + s * xh.iter().map(|t| (t - mu1).powi(2)).sum::<f64>();
        // z = Q Lambda R (xh - mu0)
        let y: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.r[i * n + j] * (xh[j] - mu0)).sum::<f64>() * self.lam[i]).collect();
        let z: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.q[i * n + j] * y[j]).sum()).collect();
        let ras = 10.0 * (nf - z.iter().map(|t| (2.0 * std::f64::consts::PI * t).cos()).sum::<f64>());
        let pen: f64 = x.iter().map(|t| (t.abs() - 5.0).max(0.0).powi(2)).sum();
        a.min(b) + ras + 1e4 * pen
    }
}

/// Small-population CMA-ES local search from x with step sigma; returns
/// (best x, best f, evaluations).
fn local(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], sigma: f64, cap: usize, rng: &mut Rng, tol: f64) -> (Vec<f64>, f64, usize) {
    let n = x0.len();
    let lambda = 4 + (3.0 * (n as f64).ln()) as usize;
    let mut es = Cmaes::new(x0, sigma, lambda);
    let mut used = 0;
    let mut fit = vec![0.0; lambda];
    loop {
        es.ask(rng);
        for k in 0..lambda {
            fit[k] = f(&es.pop[k * n..(k + 1) * n]);
        }
        used += lambda;
        if es.tell(&fit).is_some() || used + lambda > cap || es.sigma < tol {
            break;
        }
    }
    (es.best_x.clone(), es.best_f, used)
}

/// Cholesky solve of an SPD system (row-major, lower triangle read).
fn chol(a: &mut [f64], b: &mut [f64], p: usize) -> bool {
    for j in 0..p {
        for k in 0..j {
            let s: f64 = (0..k).map(|l| a[j * p + l] * a[k * p + l]).sum();
            a[j * p + k] = (a[j * p + k] - s) / a[k * p + k];
        }
        let d = a[j * p + j] - (0..j).map(|l| a[j * p + l].powi(2)).sum::<f64>();
        if !(d > 0.0) {
            return false;
        }
        a[j * p + j] = d.sqrt();
    }
    for i in 0..p {
        b[i] = (b[i] - (0..i).map(|l| a[i * p + l] * b[l]).sum::<f64>()) / a[i * p + i];
    }
    for i in (0..p).rev() {
        b[i] = (b[i] - (i + 1..p).map(|l| a[l * p + i] * b[l]).sum::<f64>()) / a[i * p + i];
    }
    true
}

/// Minimizer of the least-squares full quadratic through the points, if
/// convex.
fn quad_argmin(pts: &[(Vec<f64>, f64)]) -> Option<Vec<f64>> {
    let n = pts[0].0.len();
    let p = (n + 1) * (n + 2) / 2;
    let feat = |x: &[f64]| {
        let mut v = vec![1.0];
        v.extend_from_slice(x);
        for i in 0..n {
            for j in i..n {
                v.push(x[i] * x[j]);
            }
        }
        v
    };
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    for (x, f) in pts {
        let v = feat(x);
        for i in 0..p {
            atb[i] += v[i] * f;
            for j in 0..=i {
                ata[i * p + j] += v[i] * v[j];
            }
        }
    }
    let tr: f64 = (0..p).map(|i| ata[i * p + i]).sum();
    for i in 0..p {
        ata[i * p + i] += 1e-12 * tr / p as f64;
    }
    if !chol(&mut ata, &mut atb, p) {
        return None;
    }
    let mut h = vec![0.0; n * n];
    let mut k = n + 1;
    for i in 0..n {
        for j in i..n {
            h[j * n + i] = if i == j { 2.0 * atb[k] } else { atb[k] };
            k += 1;
        }
    }
    let mut g: Vec<f64> = atb[1..=n].iter().map(|t| -t).collect();
    if !chol(&mut h, &mut g, n) {
        return None;
    }
    Some(g)
}

/// One-shot basin map: ridge regression on `m` random Fourier features of
/// bandwidth `ell` (a random-feature network whose readout is one least-
/// squares solve), fitted to the samples; returns the map's `k` best
/// distinct minima among `probe` random points (at least `sep` apart).
#[allow(clippy::too_many_arguments)]
fn basin_map(xs: &[Vec<f64>], fs: &[f64], m: usize, ell: f64, probe: usize, k: usize, sep: f64, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = xs[0].len();
    let w: Vec<f64> = (0..m * n).map(|_| rng.normal() / ell).collect();
    let b: Vec<f64> = (0..m).map(|_| 2.0 * std::f64::consts::PI * rng.uniform()).collect();
    let feat = |x: &[f64]| -> Vec<f64> {
        let mut v = vec![1.0];
        v.extend((0..m).map(|j| ((0..n).map(|i| w[j * n + i] * x[i]).sum::<f64>() + b[j]).cos()));
        v
    };
    let p = m + 1;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    let fmean = fs.iter().sum::<f64>() / fs.len() as f64;
    for (x, &f) in xs.iter().zip(fs) {
        let v = feat(x);
        for i in 0..p {
            atb[i] += v[i] * (f - fmean);
            for j in 0..=i {
                ata[i * p + j] += v[i] * v[j];
            }
        }
    }
    let tr: f64 = (0..p).map(|i| ata[i * p + i]).sum();
    for i in 0..p {
        ata[i * p + i] += 1e-3 * tr / p as f64;
    }
    if !chol(&mut ata, &mut atb, p) {
        return vec![];
    }
    let mut cand: Vec<(f64, Vec<f64>)> = (0..probe)
        .map(|_| {
            let x: Vec<f64> = (0..n).map(|_| 10.0 * rng.uniform() - 5.0).collect();
            let v = feat(&x);
            ((0..p).map(|i| v[i] * atb[i]).sum::<f64>(), x)
        })
        .collect();
    cand.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<Vec<f64>> = Vec::new();
    for (_, x) in cand {
        if out.iter().all(|y| (0..n).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt() > sep) {
            out.push(x);
            if out.len() == k {
                break;
            }
        }
    }
    out
}

/// A resumable basin-hopping chain with envelope steps.
struct Chain {
    x: Vec<f64>,
    fx: f64,
    minima: Vec<(Vec<f64>, f64)>,
    hops: usize,
    fails: usize,
}

struct Knobs {
    r: f64,
    ls_sigma: f64,
    tol: f64,
    every: usize,
}

impl Chain {
    fn start(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], k: &Knobs, cap: usize, rng: &mut Rng) -> (Self, usize) {
        let (x, fx, u) = local(f, x0, k.ls_sigma, cap, rng, k.tol);
        (Chain { minima: vec![(x.clone(), fx)], x, fx, hops: 0, fails: 0 }, u)
    }

    /// Hops until `budget` evaluations are used; returns them.
    fn advance(&mut self, f: &dyn Fn(&[f64]) -> f64, k: &Knobs, budget: usize, rng: &mut Rng) -> usize {
        let n = self.x.len();
        let pq = (n + 1) * (n + 2) / 2;
        let mut used = 0;
        while used < budget {
            self.hops += 1;
            let (y, fy, u) = if self.hops.is_multiple_of(k.every) && self.minima.len() >= 2 * pq {
                self.minima.sort_by(|a, b| a.1.total_cmp(&b.1));
                self.minima.truncate(4 * pq);
                let m = self.minima.len().min(3 * pq);
                match quad_argmin(&self.minima[..m]) {
                    Some(xq) => {
                        let xq: Vec<f64> = xq.iter().map(|t| t.clamp(-5.0, 5.0)).collect();
                        local(f, &xq, 0.05, budget - used, rng, k.tol)
                    }
                    None => continue,
                }
            } else {
                let rr = k.r * (0.02f64 / k.r).powf(rng.uniform());
                let y: Vec<f64> = self.x.iter().map(|t| (t + rr * (2.0 * rng.uniform() - 1.0)).clamp(-5.0, 5.0)).collect();
                local(f, &y, k.ls_sigma, budget - used, rng, k.tol)
            };
            used += u;
            self.minima.push((y.clone(), fy));
            if fy < self.fx - 1e-9 {
                (self.x, self.fx) = (y, fy);
                self.fails = 0;
            } else {
                self.fails += 1;
            }
        }
        used
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).map_or(5, |s| s.parse().unwrap());
    let budget: usize = a.get(2).map_or(500_000, |s| s.parse().unwrap());
    let r: f64 = a.get(3).map_or(0.6, |s| s.parse().unwrap());
    let runs: u64 = a.get(4).map_or(10, |s| s.parse().unwrap());
    let env = |k: &str, d: f64| std::env::var(k).map_or(d, |v| v.parse().unwrap());
    let k = Knobs { r, ls_sigma: env("LS_SIGMA", 0.2), tol: env("LS_TOL", 1e-3), every: env("EVERY", 10.0) as usize };
    let map_n = env("MAP", 5000.0) as usize;
    let map_ell = env("MAP_ELL", 4.0);
    let cands = env("CANDS", 4.0) as usize;
    let round0 = env("ROUND0", 10000.0) as usize;
    let stall = env("STALL", 60.0) as usize;
    let mut solved = 0;
    let mut hit_evals = Vec::new();
    for run in 0..runs {
        let prob = Lunacek::new(n, 1000 + run);
        let f = |x: &[f64]| prob.f(x);
        let mut rng = Rng::new(run + 7);
        let mut used = 0;
        let mut best = f64::INFINITY;
        let mut hit = None;
        let track = |fx: f64, used: usize, best: &mut f64, hit: &mut Option<usize>, x: &[f64], rng: &mut Rng| -> usize {
            // Polish a promising point to full accuracy.
            let mut extra = 0;
            if fx < *best {
                *best = fx;
                if fx < 1e-2 && hit.is_none() {
                    let (_, fp, u) = local(&f, x, 1e-2, 50_000, rng, 1e-10);
                    extra = u;
                    *best = best.min(fp);
                }
                if *best < 1e-8 && hit.is_none() {
                    *hit = Some(used + extra);
                }
            }
            extra
        };
        while used < budget && hit.is_none() {
            // Map: candidate funnels from one random-feature fit.
            let starts: Vec<Vec<f64>> = if map_n > 0 && used + map_n < budget {
                let xs: Vec<Vec<f64>> = (0..map_n).map(|_| (0..n).map(|_| 10.0 * rng.uniform() - 5.0).collect()).collect();
                let fs: Vec<f64> = xs.iter().map(|x| f(x)).collect();
                used += map_n;
                basin_map(&xs, &fs, 300, map_ell, 20000, cands, 2.0, &mut rng)
            } else {
                (0..cands).map(|_| (0..n).map(|_| 10.0 * rng.uniform() - 5.0).collect()).collect()
            };
            let mut chains: Vec<Chain> = Vec::new();
            for x0 in &starts {
                let (c, u) = Chain::start(&f, x0, &k, budget - used, &mut rng);
                used += u;
                chains.push(c);
            }
            // Race by successive halving, then full throttle on the winner
            // until it stalls.
            let mut round = round0;
            while used < budget && hit.is_none() {
                for c in chains.iter_mut() {
                    let u = c.advance(&f, &k, round.min(budget - used), &mut rng);
                    used += u;
                    let (fx, x) = (c.fx, c.x.clone());
                    used += track(fx, used, &mut best, &mut hit, &x, &mut rng);
                    if used >= budget || hit.is_some() {
                        break;
                    }
                }
                chains.sort_by(|a, b| a.fx.total_cmp(&b.fx));
                if chains.len() > 1 {
                    chains.truncate(chains.len().div_ceil(2));
                    round *= 2;
                } else if chains[0].fails > stall {
                    break;
                }
            }
        }
        if let Some(h) = hit {
            solved += 1;
            hit_evals.push(h);
        }
        if std::env::var_os("VERBOSE").is_some() {
            println!("run {run}: best {best:.3e} hit {hit:?}");
        }
    }
    hit_evals.sort();
    println!("n={n} budget={budget}: solved {solved}/{runs}, evals to hit {hit_evals:?}");
}
