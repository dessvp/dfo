//! Funnel race: a regime for multi-funnel landscapes (a smooth "big
//! valley" structure covered by ripples, such as BBOB f24, the Lunacek
//! bi-Rastrigin function), where CMA-ES with large populations follows the
//! widest funnel and local searches cannot tell funnels apart.
//!
//! 1. Map: a one-shot random-feature regression (ridge readout on random
//!    Fourier features of low bandwidth) on uniform samples smooths out the
//!    ripples; its distinct minima are candidate funnels. The map locates
//!    funnels but cannot rank them (a wider funnel has a lower smoothed
//!    mean), so it is not trusted for that.
//! 2. Race: a monotonic basin-hopping chain in each candidate (perturb by
//!    about one ripple, small-population CMA-ES local search, accept if
//!    better), with successive halving: the better half by best local
//!    minimum continues with twice the budget, until one chain is left.
//! 3. Envelope steps: every few hops a chain fits a full quadratic to its
//!    best local minima, which lie on the funnel's smooth envelope, and
//!    searches at its minimizer: random hops rarely hit the right cell
//!    near a flat funnel bottom.
//!
//! Basin hopping: Leary (2000), Grosso, Locatelli & Schoen (2007);
//! random features: Rahimi & Recht (2007).

use dfo_cmaes::cmaes::{Cmaes, Rng};

/// Candidate funnels raced per map.
const CANDIDATES: usize = 4;
/// Map samples and first race round, per dimension.
const MAP_PER_N: usize = 1000;
const ROUND_PER_N: usize = 1000;
/// Random features of the map.
const FEATURES: usize = 300;
/// Relative to the narrowest box width: map bandwidth, candidate
/// separation, largest and smallest hop, local-search step and tolerance.
const MAP_ELL: f64 = 0.4;
const SEPARATION: f64 = 0.2;
const HOP_MAX: f64 = 0.06;
const HOP_MIN: f64 = 0.002;
const LS_SIGMA: f64 = 0.02;
const LS_TOL: f64 = 1e-4;
/// Hops between envelope steps; failed hops after which the last chain
/// is abandoned for a new map.
const EVERY: usize = 10;
const STALL: usize = 60;

struct Ctx<'a> {
    u: &'a [f64],
    v: &'a [f64],
    width: f64,
}

#[allow(clippy::too_many_arguments)]
/// Small-population CMA-ES from `x0` with step `sigma` until its step falls
/// below `tol` (or it stops, or `cap` evaluations). Returns the best point,
/// its value and the evaluations used.
fn local<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    c: &Ctx,
    x0: &[f64],
    sigma: f64,
    tol: f64,
    cap: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64, usize) {
    let n = x0.len();
    let lambda = 4 + (3.0 * (n as f64).ln()) as usize;
    let mut es = Cmaes::new(x0, sigma, lambda);
    let mut fit = vec![0.0; lambda];
    let mut y = vec![0.0; n];
    let mut used = 0;
    while used + lambda <= cap && !done() {
        es.ask(rng);
        for k in 0..lambda {
            // Evaluate inside the box; CMA-ES sees the clipped point's value.
            for i in 0..n {
                y[i] = es.pop[k * n + i].clamp(c.u[i], c.v[i]);
            }
            fit[k] = f(&y);
            es.observe(&y, fit[k]);
        }
        used += lambda;
        if es.tell_ranks(&fit).is_some() || es.sigma < tol {
            break;
        }
    }
    (es.best_x.clone(), es.best_f, used)
}

/// Cholesky solve (row-major, lower triangle read); false if not SPD.
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

/// Minimizer of the least-squares full quadratic through `pts`, if convex.
fn quad_argmin(pts: &[(Vec<f64>, f64)]) -> Option<Vec<f64>> {
    let n = pts[0].0.len();
    let p = (n + 1) * (n + 2) / 2;
    let mut v = Vec::with_capacity(p);
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    // Centred on the best point for conditioning.
    let c = &pts[0].0;
    for (x, f) in pts {
        v.clear();
        v.push(1.0);
        v.extend(x.iter().zip(c).map(|(a, b)| a - b));
        for i in 0..n {
            for j in i..n {
                v.push((x[i] - c[i]) * (x[j] - c[j]));
            }
        }
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
    Some(g.iter().zip(c).map(|(d, b)| d + b).collect())
}

/// The map's `k` best distinct minima (at least `sep` apart), from a ridge
/// fit on random Fourier features of bandwidth `ell`.
#[allow(clippy::too_many_arguments)]
fn basin_map(xs: &[Vec<f64>], fs: &[f64], c: &Ctx, k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = xs[0].len();
    let (m, ell, sep) = (FEATURES, MAP_ELL * c.width, SEPARATION * c.width);
    let w: Vec<f64> = (0..m * n).map(|_| rng.normal() / ell).collect();
    let b: Vec<f64> = (0..m).map(|_| 2.0 * std::f64::consts::PI * rng.uniform()).collect();
    let feat = |x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.push(1.0);
        out.extend((0..m).map(|j| ((0..n).map(|i| w[j * n + i] * x[i]).sum::<f64>() + b[j]).cos()));
    };
    let p = m + 1;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    let finite: Vec<f64> = fs.iter().copied().filter(|t| t.is_finite()).collect();
    let fmean = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
    let mut phi = Vec::with_capacity(p);
    for (x, &f) in xs.iter().zip(fs) {
        if !f.is_finite() {
            continue;
        }
        feat(x, &mut phi);
        for i in 0..p {
            atb[i] += phi[i] * (f - fmean);
            for j in 0..=i {
                ata[i * p + j] += phi[i] * phi[j];
            }
        }
    }
    let tr: f64 = (0..p).map(|i| ata[i * p + i]).sum();
    for i in 0..p {
        ata[i * p + i] += 1e-3 * tr / p as f64;
    }
    if !chol(&mut ata, &mut atb, p) {
        return Vec::new();
    }
    let mut cand: Vec<(f64, Vec<f64>)> = (0..20 * MAP_PER_N.min(1000))
        .map(|_| {
            let x: Vec<f64> = (0..n).map(|i| c.u[i] + rng.uniform() * (c.v[i] - c.u[i])).collect();
            feat(&x, &mut phi);
            ((0..p).map(|i| phi[i] * atb[i]).sum::<f64>(), x)
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

/// A resumable basin-hopping chain.
struct Chain {
    x: Vec<f64>,
    fx: f64,
    minima: Vec<(Vec<f64>, f64)>,
    hops: usize,
    fails: usize,
}

impl Chain {
    /// Hops until about `budget` evaluations are used; returns them.
    fn advance<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, done: &mut dyn FnMut() -> bool, c: &Ctx, budget: usize, rng: &mut Rng) -> usize {
        let n = self.x.len();
        let pq = (n + 1) * (n + 2) / 2;
        let w = c.width;
        let mut used = 0;
        while used < budget && !done() {
            self.hops += 1;
            let envelope = self.hops.is_multiple_of(EVERY) && self.minima.len() >= 2 * pq;
            let start: Vec<f64> = if envelope {
                self.minima.sort_by(|a, b| a.1.total_cmp(&b.1));
                self.minima.truncate(4 * pq);
                let m = self.minima.len().min(3 * pq);
                match quad_argmin(&self.minima[..m]) {
                    Some(xq) => xq.iter().enumerate().map(|(i, t)| t.clamp(c.u[i], c.v[i])).collect(),
                    None => continue,
                }
            } else {
                let r = HOP_MAX * w * (HOP_MIN / HOP_MAX).powf(rng.uniform());
                self.x.iter().enumerate().map(|(i, t)| (t + r * (2.0 * rng.uniform() - 1.0)).clamp(c.u[i], c.v[i])).collect()
            };
            let sigma = if envelope { 0.25 * LS_SIGMA * w } else { LS_SIGMA * w };
            let (y, fy, u) = local(f, done, c, &start, sigma, LS_TOL * w, budget - used, rng);
            used += u;
            if u == 0 {
                break;
            }
            self.minima.push((y.clone(), fy));
            if fy < self.fx - 1e-12 * self.fx.abs() {
                (self.x, self.fx) = (y, fy);
                self.fails = 0;
            } else {
                self.fails += 1;
            }
        }
        used
    }
}

/// One funnel race over `[u, v]` with at most `max_evals` evaluations,
/// until `done()`. Returns the evaluations used, the best point and value.
pub fn race<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    u: &[f64],
    v: &[f64],
    max_evals: usize,
    rng: &mut Rng,
) -> (usize, Vec<f64>, f64) {
    let n = u.len();
    let width = (0..n).map(|i| v[i] - u[i]).fold(f64::INFINITY, f64::min);
    let c = Ctx { u, v, width };
    let mut used = 0;
    let (mut best_x, mut best_f) = (u.to_vec(), f64::INFINITY);
    let mut polished = f64::INFINITY;
    let map_n = MAP_PER_N * n;
    while used + map_n < max_evals && !done() {
        // Map, then one chain per candidate funnel.
        let xs: Vec<Vec<f64>> = (0..map_n).map(|_| (0..n).map(|i| u[i] + rng.uniform() * (v[i] - u[i])).collect()).collect();
        let mut fs = Vec::with_capacity(map_n);
        for x in &xs {
            if done() {
                break;
            }
            fs.push(f(x));
            used += 1;
        }
        if done() {
            break;
        }
        for (x, &fx) in xs.iter().zip(&fs) {
            if fx < best_f {
                (best_x, best_f) = (x.clone(), fx);
            }
        }
        let starts = basin_map(&xs, &fs, &c, CANDIDATES, rng);
        let mut chains: Vec<Chain> = Vec::new();
        for x0 in &starts {
            let (x, fx, uu) = local(f, done, &c, x0, LS_SIGMA * width, LS_TOL * width, max_evals - used, rng);
            used += uu;
            chains.push(Chain { minima: vec![(x.clone(), fx)], x, fx, hops: 0, fails: 0 });
        }
        if chains.is_empty() {
            break;
        }
        // Successive halving, then the winner until it stalls.
        let mut round = ROUND_PER_N * n;
        while used < max_evals && !done() {
            // Less room left than a step needs: stop instead of spinning.
            let before = used;
            for ch in chains.iter_mut() {
                used += ch.advance(f, done, &c, round.min(max_evals - used), rng);
                if ch.fx < best_f {
                    (best_x, best_f) = (ch.x.clone(), ch.fx);
                }
                if used >= max_evals || done() {
                    break;
                }
            }
            // Polish the best point to full accuracy when it has improved.
            if best_f < polished && used < max_evals && !done() {
                let (x, fx, uu) = local(f, done, &c, &best_x, LS_SIGMA * width * 0.1, 1e-12 * width, (200 * n).min(max_evals - used), rng);
                used += uu;
                if fx < best_f {
                    (best_x, best_f) = (x, fx);
                }
                polished = best_f;
            }
            if used == before {
                return (used, best_x, best_f);
            }
            chains.sort_by(|a, b| a.fx.total_cmp(&b.fx));
            if chains.len() > 1 {
                chains.truncate(chains.len().div_ceil(2));
                round *= 2;
            } else if chains[0].fails > STALL {
                break;
            }
        }
    }
    (used, best_x, best_f)
}
