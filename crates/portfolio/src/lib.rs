//! Derivative-free global optimization portfolio for bound-constrained
//! black-box problems.
//!
//! 1. MCS (with BOBYQA local searches) runs first for a short budget: its
//!    initialization list and first sweeps make the fastest early progress
//!    (from 20-D only its initialization).
//! 2. A separability test at MCS's best point; on separable functions up
//!    to 5-D, a coordinate-wise solver (1-D Brent passes, then a grid pass
//!    for multimodal coordinates) runs before anything else.
//! 3. From 5-D, a bound-constrained BFGS phase with finite-difference
//!    gradients; its inverse Hessian and step length warm-start the first
//!    CMA-ES run, which caps that covariance's condition number after a
//!    while (the inverse Hessian is wrong on non-smooth functions).
//! 4. Then restart slots go to whichever regime has used the fewest
//!    evaluations per unit weight (BIPOP's balancing, extended):
//!    - `L`: lq-CMA-ES with large populations (IPOP doubling); the first
//!      run starts at the best point so far with MCS's archive as model
//!      data;
//!    - `S`: lq-CMA-ES with small random populations and step sizes
//!      (BIPOP's local regime);
//!    - `M`: the same MCS run, resumed: its partition keeps covering the
//!      box (the convergence guarantee of MCS) and finds isolated basins
//!      (up to 4-D);
//!    - `D`: L-SHADE, from 10^4·n evaluations in 6-20-D;
//!    - `F`: the funnel race of [`funnel`] for multi-funnel landscapes,
//!      from 10^4·n evaluations up to 5-D.
//!
//! MCS is resumable (`dfo_mcs::Mcs::run` stops after whole steps and
//! continues later), so the portfolio pauses and resumes it without
//! threads; every evaluation happens on the caller's thread.

pub mod de;
pub mod funnel;
pub mod qn;

use dfo_cmaes::cmaes::{Cmaes, Rng, default_lambda};
use dfo_cmaes::lq::{Archive, generation};
use dfo_mcs::{Config as McsConfig, Mcs, Status};

/// Settings of the portfolio.
#[derive(Clone, Debug)]
pub struct HybridConfig {
    /// Evaluations of the initial MCS phase, per dimension (None:
    /// `default_mcs_first`).
    pub mcs_first: Option<f64>,
    /// Initial CMA-ES step size as a fraction of the box width.
    pub sigma_frac: f64,
    /// Use the resumed-MCS regime in the restart phase.
    pub mcs_regime: bool,
    /// Use BIPOP's small-population regime.
    pub small_regime: bool,
    /// Test for separability and solve separable functions coordinate-wise.
    pub separable: bool,
    /// Budget weights of the L, S and M regimes (`None`: by dimension, see
    /// `default_weights`): each slot goes to the
    /// regime with the smallest evaluations / weight.
    pub weights: Option<[f64; 3]>,
    /// Scale the weights online by each regime's success rate (runs that
    /// improved the best value so far), as NBIPOP-aCMA-ES moves budget to
    /// the regime that finds the best solutions.
    pub adaptive: bool,
    /// Lower bound of MCS's initial local-search radius, as a fraction of
    /// the narrowest bound width.
    pub mcs_rho_min: f64,
    /// Budget of the quasi-Newton (BFGS) phase after MCS, in evaluations
    /// per dimension (0 = off; `None` = [`default_qn`]). The first CMA-ES
    /// run is warm-started from its best point, inverse Hessian and step
    /// length.
    pub qn: Option<f64>,
    /// NIPOP: each large-population restart divides the initial step size
    /// by this factor (Loshchilov 2013 uses 1.6; 1 = off), down to
    /// `sigma0 * SIGMA_FLOOR`.
    pub sigma_decay: f64,
    /// Initial inverse Hessian of the quasi-Newton phase, relative to unit
    /// curvature on a box of width 10 (None: `default_qn_h0`).
    pub qn_h0: Option<f64>,
    /// Whether the first CMA-ES run inherits the quasi-Newton step length
    /// and inverse Hessian (otherwise it only starts at its best point).
    pub qn_warm: bool,
    /// NBIPOP: the regime that found the best solution so far gets twice
    /// the budget weight of the others.
    pub best_boost: bool,
    /// Tabu restarts: a restart that would start at a uniform random point
    /// draws this many candidates and starts at the one farthest (in the
    /// box-normalized metric) from every earlier run's best point
    /// (0 = plain uniform start).
    pub tabu: usize,
    pub seed: u64,
    /// Initial step size of the large-population restarts after the first
    /// run, as a fraction of the narrowest box width (None: `sigma_frac`).
    pub sigma_restart: Option<f64>,
    /// On separable functions, solve coordinate-wise before the other
    /// phases as well as after the first CMA-ES run (None: up to
    /// `SEP_FIRST_MAX_N` dimensions), with `sep_grid` grid points per
    /// coordinate.
    pub sep_first: Option<bool>,
    pub sep_grid: usize,
    /// Budget weight of the L-SHADE regime (None: `default_de_weight`),
    /// and the evaluations per dimension before it may run.
    pub de_weight: Option<f64>,
    pub de_start: f64,
    /// Budget weight of the funnel-race regime (None: `default_funnel_weight`), and the
    /// evaluations per dimension before it may run.
    pub funnel_weight: Option<f64>,
    pub funnel_start: f64,
    /// Budget of the first race per dimension (None: `FUNNEL_FIRST`).
    pub funnel_first: Option<f64>,
    /// The first CMA-ES run, warm-started from the quasi-Newton inverse
    /// Hessian, caps the condition number of its covariance at `warm_cond`
    /// after `warm_cap`·n evaluations (0: never). The inverse Hessian
    /// speeds up the run on smooth ill-conditioned functions, but on
    /// non-smooth ones it is wrong (f6 in 20-D: 32k evaluations to
    /// 1e-8 instead of 10k cold). Development instances, 20-D f6-f14:
    /// 1000·D 89.2 -> 96.1 (f6 42 -> 100), 100·D unchanged.
    pub warm_cap: f64,
    pub warm_cond: f64,
    /// Budget weight of quasi-Newton restarts from random points (0: off),
    /// and the budget of each per dimension.
    pub qn_weight: f64,
    pub qn_restart: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig {
            mcs_first: None,
            sigma_frac: 0.1,
            mcs_regime: true,
            small_regime: true,
            separable: true,
            weights: None,
            adaptive: false,
            mcs_rho_min: 0.02,
            qn: None,
            sigma_decay: 1.0,
            qn_h0: None,
            qn_warm: true,
            best_boost: false,
            tabu: 0,
            seed: 1,
            sigma_restart: None,
            sep_first: None,
            sep_grid: 81,
            de_weight: None,
            de_start: 10000.0,
            funnel_weight: None,
            funnel_start: 10000.0,
            funnel_first: None,
            warm_cap: 200.0,
            warm_cond: 100.0,
            qn_weight: 0.0,
            qn_restart: 50.0,
        }
    }
}

/// Smallest NIPOP step size, as a fraction of the default one.
const SIGMA_FLOOR: f64 = 0.01;

/// Coordinate-wise solving comes first on separable functions up to this
/// dimension. Development instances, 1000·D: +1.6 at 100·D in 2-D, +0.5
/// in 3-D, +0.2 in 5-D, -0.6 in 10-D (where a robust grid pass no longer
/// fits in 100 evaluations per coordinate).
const SEP_FIRST_MAX_N: usize = 5;

/// Grid points per coordinate of the coordinate-wise pass after the first
/// CMA-ES run: dense, as by then a robust pass matters more than its cost.
const FALLBACK_GRID: usize = 201;

/// Quasi-Newton budget per dimension: 100 from 10-D up, 30 from 5-D, off
/// below. Mean score over budgets up to 1e4·D, three seeds on the tuning
/// instances: 20-D 51.0 -> 53.2 (100·D column 23.2 -> 34.7), 10-D 59.0 ->
/// 59.5 (37.6 -> 42.6). In 5-D a short phase pays (all five budgets, 30:
/// 67.1 -> 67.7, 100·D 49.4 -> 50.3; 100: 100·D 49.6), in 3-D it loses
/// (100·D 55.4 -> 53.7): MCS and the full-quadratic lq-CMA-ES already
/// cover the smooth functions there.
pub fn default_qn(n: usize) -> f64 {
    if n >= 10 {
        100.0
    } else if n >= 5 {
        30.0
    } else {
        0.0
    }
}

/// Budget of the first funnel race, per dimension.
const FUNNEL_FIRST: usize = 60_000;

/// Weight of the funnel-race regime by dimension (0: off). Development
/// instances, from 10^4·n evaluations: 10^5·D score 98.1 -> 99.7 (2-D),
/// 97.3 -> 99.9 (3-D), 96.6 -> 99.7 (5-D), nothing lost at smaller
/// budgets; no gain in 10-D.
pub fn default_funnel_weight(n: usize) -> f64 {
    if n <= 5 { 3.0 } else { 0.0 }
}

/// Budget of the first L-SHADE run, per dimension.
const DE_FIRST: usize = 1000;

/// Weight of the L-SHADE regime by dimension (0: off). Development
/// instances, from 10^4·n evaluations: 10-D 10^5·D score 92.9 -> 94.7,
/// nothing lost at smaller budgets; no gain over the funnel race below.
/// In 20-D, with the even split of `default_weights` from 10^4·n:
/// 10^5·D on f15-f24 67.0 -> 71.0 (f21 67 -> 84, f22 28 -> 53); in 40-D
/// it was within noise on four instances.
pub fn default_de_weight(n: usize) -> f64 {
    if (6..=20).contains(&n) { 1.0 } else { 0.0 }
}

/// Regime weights by dimension: resumed MCS pays off only in low
/// dimensions, where its space-filling search is efficient (ablation on
/// BBOB f15-f24: +3.6 in 2-D, +1.6 in 3-D, -0.9 in 5-D, -1.8 in 10-D);
/// above that the restarts split evenly as in BIPOP, and from 20-D the
/// large-population regime gets twice the budget (20-D, development
/// instances: 1000·D 66.5 -> 67.6, mostly f7 36 -> 63; 10^4·D on f15-f24
/// within noise). In 20-D the split turns even after 10^4·n evaluations
/// (`used`), when the small populations pay off on f21-f22 (see
/// `default_de_weight`).
pub fn default_weights(n: usize, used: usize) -> [f64; 3] {
    if n <= 4 {
        [2.0, 1.0, 1.0]
    } else if n < 20 || (n <= FULL_RESTARTS_MAX_N && used as f64 >= 1e4 * n as f64) {
        [1.0, 1.0, 0.0]
    } else {
        [2.0, 1.0, 0.0]
    }
}

/// Initial inverse Hessian scale of the quasi-Newton phase: smaller above
/// 20-D, where unit curvature overshoots so far that backtracking eats the
/// budget (40-D, development instances: 100·D 29.6 -> 31.8, 1000·D 60.3
/// -> 61.1, 10·D 14.5 -> 13.7; in 20-D 100·D 37.3 -> 36.5). On rotated
/// Rosenbrock from the origin, 40-D: 146·D to 1e-8 at 1, 130·D at 0.1.
pub fn default_qn_h0(n: usize) -> f64 {
    if n > FULL_RESTARTS_MAX_N { 0.1 } else { 1.0 }
}

/// Initial MCS evaluations per dimension: from 20-D only its
/// initialization (about 2n evaluations), so the quasi-Newton phase starts
/// sooner (20-D: 10·D 14.0 -> 14.9, 100·D 34.7 -> 36.7; 40-D: 13.6 ->
/// 14.7 and 25.7 -> 28.7); in 10-D that loses at 100·D (43.8 -> 42.9).
pub fn default_mcs_first(n: usize) -> f64 {
    if n >= 20 { 2.1 } else { 10.0 }
}

/// The resumable MCS search, fed into the portfolio's archive.
struct McsRunner {
    m: Mcs,
    finished: bool,
}

impl McsRunner {
    fn start(u: &[f64], v: &[f64], ftarget: f64, rho_min_frac: f64) -> Self {
        let n = u.len();
        let mut cfg = McsConfig::new(n);
        cfg.f_target = ftarget;
        cfg.rho_min_frac = rho_min_frac;
        McsRunner { m: Mcs::new(u, v, cfg), finished: false }
    }

    /// Runs MCS for about `quota` evaluations: it starts no new step after
    /// that, but finishes a step in progress (an initialization, a split, a
    /// local search) within the `room` evaluations left in the whole run.
    /// Returns how many it used.
    fn run<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, quota: usize, room: usize, ar: &mut Archive, done: &mut dyn FnMut() -> bool) -> usize {
        if self.finished || quota == 0 || done() {
            return 0;
        }
        let before = self.m.evaluations();
        let mut g = |x: &[f64]| {
            let fx = f(x);
            ar.push(x, fx);
            fx
        };
        let st = self.m.run(&mut g, quota, before + room);
        let used = self.m.evaluations() - before;
        // Nothing done under the whole remaining room: the next step will
        // never fit.
        if st == Status::Finished || (st == Status::Quota && used == 0) {
            self.finished = true;
        }
        used
    }
}

/// Tests additive separability at `x` (value `fx`) on two random pairs of
/// coordinates: for a separable `f`, f(x+a e_i+b e_j) - f(x+a e_i) -
/// f(x+b e_j) + f(x) vanishes up to rounding. Returns (separable, used).
#[allow(clippy::too_many_arguments)]
fn separable<F: FnMut(&[f64]) -> f64>(f: &mut F, x: &[f64], fx: f64, u: &[f64], v: &[f64], rng: &mut Rng, ar: &mut Archive) -> (bool, usize) {
    let n = x.len();
    if n < 2 {
        return (true, 0);
    }
    let mut used = 0;
    let mut ok = true;
    let mut y = x.to_vec();
    let eval = |y: &[f64], f: &mut F, ar: &mut Archive, used: &mut usize| {
        let fy = f(y);
        ar.push(y, fy);
        *used += 1;
        fy
    };
    for _ in 0..2 {
        let i = (rng.uniform() * n as f64) as usize % n;
        let j = (i + 1 + (rng.uniform() * (n - 1) as f64) as usize % (n - 1)) % n;
        // Steps inside the box, of 10-30% of the width.
        let step = |k: usize, rng: &mut Rng| {
            let w = (0.1 + 0.2 * rng.uniform()) * (v[k] - u[k]);
            if x[k] + w <= v[k] { w } else { -w }
        };
        let (a, b) = (step(i, rng), step(j, rng));
        y[i] = x[i] + a;
        let fi = eval(&y, f, ar, &mut used);
        y[j] = x[j] + b;
        let fij = eval(&y, f, ar, &mut used);
        y[i] = x[i];
        let fj = eval(&y, f, ar, &mut used);
        y[j] = x[j];
        let scale = fij.abs().max(fi.abs()).max(fj.abs()).max(fx.abs()).max(1e-300);
        if !((fij - fi - fj + fx).abs() <= 1e-9 * scale) {
            ok = false;
            break;
        }
    }
    (ok, used)
}

/// Brent's derivative-free 1-D minimization of `g` on the bracket `[a, c]`
/// from its best known point `b` (value `fb`), to absolute tolerance `tol`
/// or at most `cap` evaluations. Returns the best point and value and the
/// evaluations used. Golden-section and parabolic steps as in Brent's
/// procedure `localmin` (Brent 1973, ch. 5), with its variable names.
fn brent<G: FnMut(f64) -> f64>(g: &mut G, mut a: f64, mut c: f64, b: f64, fb: f64, tol: f64, cap: usize) -> (f64, f64, usize) {
    const GOLD: f64 = 0.381_966_011_250_105;
    let (mut x, mut w, mut v) = (b, b, b);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let (mut d, mut e) = (0.0f64, 0.0f64);
    let mut used = 0;
    while used < cap {
        let m = 0.5 * (a + c);
        let tol1 = 1e-12 * x.abs() + tol;
        let t2 = 2.0 * tol1;
        if (x - m).abs() <= t2 - 0.5 * (c - a) {
            break;
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // Parabola through x, w, v.
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let r = e;
            e = d;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (c - x) {
                d = p / q;
                let t = x + d;
                if t - a < t2 || c - t < t2 {
                    d = if x < m { tol1 } else { -tol1 };
                }
                golden = false;
            }
        }
        if golden {
            e = if x < m { c - x } else { a - x };
            d = GOLD * e;
        }
        let t = if d.abs() >= tol1 { x + d } else { x + tol1.copysign(d) };
        let ft = g(t);
        used += 1;
        if ft <= fx {
            if t < x {
                c = x;
            } else {
                a = x;
            }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, t, ft);
        } else {
            if t < x {
                a = t;
            } else {
                c = t;
            }
            if ft <= fw || w == x {
                (v, fv, w, fw) = (w, fw, t, ft);
            } else if ft <= fv || v == x || v == w {
                (v, fv) = (t, ft);
            }
        }
    }
    (x, fx, used)
}

/// One pass of 1-D local minimizations, coordinate by coordinate, from
/// `x` (bracketing, then Brent; at most `PER` evaluations per coordinate):
/// solves separable unimodal functions in O(n) evaluations. Evaluated
/// points go into `ar`. Updates `x`, returns (fx, used).
#[allow(clippy::too_many_arguments)]
fn coordinate_descent<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    x: &mut [f64],
    mut fx: f64,
    u: &[f64],
    v: &[f64],
    budget: usize,
    ar: &mut Archive,
) -> (f64, usize) {
    const PER: usize = 30;
    let n = x.len();
    let mut used = 0;
    for i in 0..n {
        let cap = (used + PER).min(budget);
        if used + 2 > cap || done() {
            break;
        }
        let (lo, hi) = (u[i], v[i]);
        let x0 = x[i];
        let mut y = x.to_vec();
        let mut g = |t: f64| {
            if done() {
                return f64::INFINITY;
            }
            y[i] = t;
            let ft = f(&y);
            ar.push(&y, ft);
            ft
        };
        let mut k = 0;
        // Bracket: a < b < c with f(b) <= f(a), f(c), or b on a bound.
        let h = 0.05 * (hi - lo);
        let up = (x0 + h).min(hi);
        let fup = g(up);
        k += 1;
        let (a, b, c, fb) = if fup < fx {
            expand(&mut g, x0, up, fup, lo, hi, cap - used, &mut k)
        } else {
            let dn = (x0 - h).max(lo);
            let fdn = g(dn);
            k += 1;
            if fdn < fx { expand(&mut g, x0, dn, fdn, lo, hi, cap - used, &mut k) } else { (dn, x0, up, fx) }
        };
        let left = (cap - used).saturating_sub(k);
        let (xb, fb, k2) = brent(&mut g, a, c, b, fb, 1e-10 * (hi - lo), left);
        used += k + k2;
        x[i] = xb;
        fx = fb;
        if used >= budget || done() {
            break;
        }
    }
    (fx, used)
}

/// Steps downhill from `x0` through `x1` (value `f1`), growing by the golden
/// ratio, until `g` rises or a bound is reached; returns a bracket
/// (a, b, c, f(b)). Counts evaluations in `k`, at most `cap` in total.
#[allow(clippy::too_many_arguments)]
fn expand<G: FnMut(f64) -> f64>(g: &mut G, x0: f64, x1: f64, f1: f64, lo: f64, hi: f64, cap: usize, k: &mut usize) -> (f64, f64, f64, f64) {
    let (mut prev, mut x, mut fx) = (x0, x1, f1);
    loop {
        let t = (x + 1.618 * (x - prev)).clamp(lo, hi);
        if t == x || *k + 1 > cap {
            return (prev.min(x), x, prev.max(x), fx);
        }
        let ft = g(t);
        *k += 1;
        if ft >= fx {
            return (prev.min(t), x, prev.max(t), fx);
        }
        (prev, x, fx) = (x, t, ft);
    }
}

/// Minimizes a separable `f` one coordinate at a time from `x` (value
/// `fx`): a `grid`-point grid over the coordinate's range, a short Brent
/// search into each of the `REFINE` best grid minima (the grid alone cannot
/// rank neighbouring basins of an oscillating function), then Brent to full
/// accuracy in the best basin. Evaluated points go into `ar`. Updates `x`,
/// returns (fx, used).
#[allow(clippy::too_many_arguments)]
fn coordinatewise<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    x: &mut [f64],
    mut fx: f64,
    u: &[f64],
    v: &[f64],
    budget: usize,
    ar: &mut Archive,
    grid: usize,
) -> (f64, usize) {
    const REFINE: usize = 6;
    const COARSE_EVALS: usize = 6;
    const FINE_EVALS: usize = 30;
    let n = x.len();
    let mut used = 0;
    let mut fg = vec![0.0; grid];
    for i in 0..n {
        let (lo, hi) = (u[i], v[i]);
        let h = (hi - lo) / (grid - 1) as f64;
        let mut y = x.to_vec();
        let mut count = 0usize;
        let mut g = |t: f64| {
            if done() || used + count >= budget {
                return f64::INFINITY;
            }
            y[i] = t;
            let ft = f(&y);
            ar.push(&y, ft);
            count += 1;
            ft
        };
        let (mut bi, mut bf) = (x[i], fx);
        for (k, fk) in fg.iter_mut().enumerate() {
            *fk = g(lo + k as f64 * h);
            if *fk < bf {
                (bi, bf) = (lo + k as f64 * h, *fk);
            }
        }
        // Grid minima, best first; a coarse search in each.
        let mut minima: Vec<usize> =
            (0..grid).filter(|&k| fg[k].is_finite() && (k == 0 || fg[k] <= fg[k - 1]) && (k + 1 == grid || fg[k] <= fg[k + 1])).collect();
        minima.sort_by(|&a, &b| fg[a].total_cmp(&fg[b]));
        let coarse = 1e-3 * h;
        for &k in minima.iter().take(REFINE) {
            let t = lo + k as f64 * h;
            let (xb, fb, _) = brent(&mut g, (t - h).max(lo), (t + h).min(hi), t, fg[k], coarse, COARSE_EVALS);
            if fb < bf {
                (bi, bf) = (xb, fb);
            }
        }
        // Full accuracy in the best basin (its grid cell on either side).
        let (xb, fb, _) = brent(&mut g, (bi - h).max(lo), (bi + h).min(hi), bi, bf, 1e-10 * (hi - lo), FINE_EVALS);
        if fb < bf {
            (bi, bf) = (xb, fb);
        }
        used += count;
        x[i] = bi;
        fx = bf;
        if used >= budget || done() {
            break;
        }
    }
    (fx, used)
}

/// Up to this dimension the first CMA-ES run starts at the box centre
/// rather than at MCS's best point, still with MCS's points as model
/// data. 2-D, development instances: 100·D 62.0 -> 65.1, 1000·D 90.4 ->
/// 91.3, 10^4·D 97.1 -> 96.9; in 3-D it loses (1000·D 87.5 -> 85.5).
const CENTRE_FIRST_MAX_N: usize = 2;

/// Restarts above this dimension fit diagonal quadratic surrogates only.
/// The full model has (n+1)(n+2)/2 coefficients: in 40-D each fit is a
/// 861-coefficient least-squares problem on 1300 points (about 8 ms on the
/// AMX), and on the multimodal functions that need restarts the model keeps
/// ranking well enough to stay on, so a 4e6-evaluation run took about an
/// hour. The first run, where the full model pays (ill-conditioned f10-f14),
/// keeps it.
const FULL_RESTARTS_MAX_N: usize = 20;

fn restart_archive(n: usize) -> Archive {
    let mut ar = Archive::default();
    ar.diagonal_only = n > FULL_RESTARTS_MAX_N;
    ar
}

/// One complete lq-CMA-ES run; returns the evaluations used. A run
/// started from a quasi-Newton inverse Hessian (`warm`) caps the condition
/// number of its covariance at `cap.1` after `cap.0`·n evaluations.
#[allow(clippy::too_many_arguments)]
fn cma_run<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    x0: &[f64],
    sigma: f64,
    lambda: usize,
    budget: usize,
    rng: &mut Rng,
    ar: &mut Archive,
    warm: Option<&[f64]>,
    cap: Option<(f64, f64)>,
) -> (usize, Vec<f64>) {
    let mut es = Cmaes::new(x0, sigma, lambda);
    if let Some(c) = warm {
        es.set_covariance(c);
    }
    let mut used = 0;
    let mut gate = dfo_cmaes::lq::Gate::default();
    let late = warm.and(cap).map(|(k, c)| ((k * x0.len() as f64) as usize, c));
    let mut capped = false;
    while used < budget && !done() {
        if let Some((k, c)) = late
            && !capped
            && used >= k
        {
            es.cap_condition(c);
            capped = true;
        }
        let (u, stop) = generation(&mut es, ar, f, rng, budget - used, &mut gate);
        used += u;
        if stop {
            break;
        }
    }
    (used, es.best_x)
}

/// Uniform start in `[u, v]`; with `k > 0`, the best of `k` candidates by
/// distance to the nearest point of `tabu`.
fn restart_point(rng: &mut Rng, u: &[f64], v: &[f64], tabu: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = u.len();
    let draw = |rng: &mut Rng| -> Vec<f64> { (0..n).map(|i| u[i] + rng.uniform() * (v[i] - u[i])).collect() };
    if k == 0 || tabu.is_empty() {
        return draw(rng);
    }
    let gap = |x: &[f64]| tabu.iter().map(|t| (0..n).map(|i| ((x[i] - t[i]) / (v[i] - u[i])).powi(2)).sum::<f64>()).fold(f64::INFINITY, f64::min);
    let mut best = draw(rng);
    let mut best_gap = gap(&best);
    for _ in 1..k {
        let x = draw(rng);
        let g = gap(&x);
        if g > best_gap {
            best = x;
            best_gap = g;
        }
    }
    best
}

/// Minimizes `f` over the box `[u, v]` until `done()` or `max_evals`
/// evaluations; returns the number of evaluations used. `ftarget` is passed
/// to MCS's stopping rule.
///
/// `f` is only called at points inside the box: candidates outside it are
/// projected onto it first, and the optimizers see the projection's value
/// plus `1e4 * sum(((x[i] - xc[i]) / (v[i] - u[i]))^2)` for a candidate `x`
/// with projection `xc`. A NaN value counts as `+inf`.
///
/// # Panics
///
/// If the dimension is 0, if `u` and `v` differ in length, or if a bound is
/// not finite or `u[i] >= v[i]`.
pub fn minimize<F: FnMut(&[f64]) -> f64>(
    mut f: F,
    mut done: impl FnMut() -> bool,
    u: &[f64],
    v: &[f64],
    ftarget: f64,
    max_evals: usize,
    cfg: &HybridConfig,
) -> usize {
    let n = u.len();
    assert!(n > 0, "minimize: the dimension must be at least 1");
    assert_eq!(n, v.len(), "minimize: the bounds u and v must have the same length");
    assert!((0..n).all(|i| u[i].is_finite() && v[i].is_finite() && u[i] < v[i]), "minimize: the bounds must be finite with u[i] < v[i]");
    // Projection onto the box: CMA-ES and L-SHADE sample unbounded
    // distributions.
    const PENALTY: f64 = 1e4;
    let mut xc = vec![0.0; n];
    let mut f = |x: &[f64]| {
        for i in 0..n {
            xc[i] = x[i].clamp(u[i], v[i]);
        }
        let y = f(&xc);
        let y = if y.is_nan() { f64::INFINITY } else { y };
        // A point outside is ranked by its projection's value plus a
        // penalty, so that the samplers are pulled back into the box.
        let d2: f64 = (0..n).map(|i| ((x[i] - xc[i]) / (v[i] - u[i])).powi(2)).sum();
        y + PENALTY * d2
    };
    let mut rng = Rng::new(cfg.seed);
    let width = (0..n).map(|i| v[i] - u[i]).fold(f64::INFINITY, f64::min);
    let sigma0 = cfg.sigma_frac * width;
    // MCS gets the whole budget; the portfolio decides how much it uses.
    let mut mcs_co = McsRunner::start(u, v, ftarget, cfg.mcs_rho_min);
    let mut mcs_ar = Archive::default();
    let mut used =
        mcs_co.run(&mut f, ((cfg.mcs_first.unwrap_or_else(|| default_mcs_first(n)) * n as f64) as usize).min(max_evals), max_evals, &mut mcs_ar, &mut done);
    // Separability test at MCS's best point; a separable function that the
    // first CMA-ES run does not solve is then solved coordinate-wise.
    let mut sep_pending = false;
    if let Some(k) = (0..mcs_ar.f.len()).min_by(|&a, &b| mcs_ar.f[a].total_cmp(&mcs_ar.f[b])) {
        // The test takes at most 6 evaluations.
        if cfg.separable && used + 6 <= max_evals && !done() {
            let (x, fx) = (mcs_ar.x[k].clone(), mcs_ar.f[k]);
            let (sep, u_t) = separable(&mut f, &x, fx, u, v, &mut rng, &mut mcs_ar);
            used += u_t;
            sep_pending = sep;
        }
    }
    let mut sep_start: Option<Vec<f64>> = None;
    // Best point so far, starting from the MCS and separability-test points.
    let best_seen = std::cell::RefCell::new(match (0..mcs_ar.f.len()).min_by(|&a, &b| mcs_ar.f[a].total_cmp(&mcs_ar.f[b])) {
        Some(k) => (mcs_ar.f[k], mcs_ar.x[k].clone()),
        None => (f64::INFINITY, vec![0.0; n]),
    });
    let mut f = |y: &[f64]| {
        let fy = f(y);
        let mut b = best_seen.borrow_mut();
        if fy < b.0 {
            *b = (fy, (0..n).map(|i| y[i].clamp(u[i], v[i])).collect());
        }
        fy
    };
    let (mut used_l, mut used_s, mut used_m, mut used_d) = (0usize, 0usize, used, 0usize);
    // L-SHADE regime: its weight, and the budget of its next run (doubling).
    let wd = cfg.de_weight.unwrap_or_else(|| default_de_weight(n));
    let mut de_budget = DE_FIRST * n;
    // Funnel-race regime: weight and next race budget (doubling).
    let wf = cfg.funnel_weight.unwrap_or_else(|| default_funnel_weight(n));
    let mut funnel_budget = (cfg.funnel_first.unwrap_or(FUNNEL_FIRST as f64) * n as f64) as usize;
    let mut used_f = 0usize;
    let mut used_q = 0usize;
    // Separable: coordinate-wise first (a 1-D local pass, then a grid
    // pass for multimodal coordinates), recorded in the MCS archive so the
    // later phases start from its best point.
    // (If that does not solve it, the pass is repeated after the first
    // CMA-ES run, from the best point then.)
    if sep_pending && cfg.sep_first.unwrap_or(n <= SEP_FIRST_MAX_N) && !done() && used < max_evals {
        let (fx, mut x) = best_seen.borrow().clone();
        let (fx, u1) = coordinate_descent(&mut f, &mut done, &mut x, fx, u, v, max_evals - used, &mut mcs_ar);
        used += u1;
        used_l += u1;
        if !done() && used < max_evals {
            let (_, u2) = coordinatewise(&mut f, &mut done, &mut x, fx, u, v, max_evals - used, &mut mcs_ar, cfg.sep_grid);
            used += u2;
            used_l += u2;
        }
    }
    // Quasi-Newton phase from MCS's best point; its state warm-starts the
    // first CMA-ES run.
    let mut warm: Option<(Vec<f64>, f64)> = None;
    let qn_per_n = cfg.qn.unwrap_or_else(|| default_qn(n));
    let qn_h0 = cfg.qn_h0.unwrap_or_else(|| default_qn_h0(n));
    if qn_per_n > 0.0
        && used < max_evals
        && !done()
        && let Some(k) = (0..mcs_ar.f.len()).min_by(|&a, &b| mcs_ar.f[a].total_cmp(&mcs_ar.f[b]))
    {
        let (x0, f0) = (mcs_ar.x[k].clone(), mcs_ar.f[k]);
        let budget = ((qn_per_n * n as f64) as usize).min(max_evals - used);
        let r = qn::bfgs(&mut f, &mut done, &x0, f0, u, v, budget, qn_h0, &mut mcs_ar);
        used += r.used;
        used_l += r.used;
        let sigma = if r.step > 0.0 { (r.step / (n as f64).sqrt()).clamp(1e-12 * width, sigma0) } else { sigma0 };
        if cfg.qn_warm {
            warm = Some((r.h, sigma));
        }
    }
    let lambda0 = default_lambda(n);
    let mut lambda_l = lambda0;
    let mut first = true;
    // Restart-phase runs and successes of L, S and M.
    let (mut runs, mut wins) = ([0usize; 3], [0usize; 3]);
    // Regime that found the best solution so far (L for the first run).
    let mut holder = 0usize;
    // Restarts of the large-population regime start wider (as in IPOP).
    let mut sigma_l = cfg.sigma_restart.map_or(sigma0, |r| r * width);
    // Best points of all CMA-ES runs so far, avoided by tabu restarts.
    let mut tabu: Vec<Vec<f64>> = Vec::new();
    while used < max_evals && !done() {
        let left = max_evals - used;
        let mut weights = cfg.weights.unwrap_or_else(|| default_weights(n, used));
        if cfg.adaptive {
            // Prior weight times the Laplace-smoothed success rate, floored
            // at a fifth of the largest so no regime starves (large
            // populations pay off late).
            for r in 0..3 {
                weights[r] *= (wins[r] as f64 + 1.0) / (runs[r] as f64 + 2.0);
            }
            let top = weights.iter().copied().fold(0.0, f64::max);
            for w in weights.iter_mut().filter(|w| **w > 0.0) {
                *w = w.max(0.2 * top);
            }
        }
        if cfg.best_boost {
            weights[holder] *= 2.0;
        }
        let before = best_seen.borrow().0;
        // Regime with the fewest evaluations per weight (L always first).
        let mut choice = 'L';
        let was_first = first;
        if !first {
            let [wl, ws, wm] = weights;
            let mut best = used_l as f64 / wl;
            if cfg.small_regime && (used_s as f64 / ws) < best {
                choice = 'S';
                best = used_s as f64 / ws;
            }
            if cfg.mcs_regime && !mcs_co.finished && (used_m as f64 / wm) < best {
                choice = 'M';
                best = used_m as f64 / wm;
            }
            // L-SHADE only joins once the faster regimes have had their
            // budget: it pays off at large budgets only.
            if wd > 0.0 && used as f64 >= cfg.de_start * n as f64 && (used_d as f64 / wd) < best {
                choice = 'D';
                best = used_d as f64 / wd;
            }
            if cfg.qn_weight > 0.0 && (used_q as f64 / cfg.qn_weight) < best {
                choice = 'Q';
                best = used_q as f64 / cfg.qn_weight;
            }
            if wf > 0.0 && used as f64 >= cfg.funnel_start * n as f64 && (used_f as f64 / wf) < best {
                choice = 'F';
            }
        }
        let u_now = match choice {
            'L' => {
                let (x0, mut ar) = if first {
                    // Start at MCS's best point (in 2-D at the box centre),
                    // with its evaluations as model data.
                    let k = (0..mcs_ar.f.len()).min_by(|&a, &b| mcs_ar.f[a].total_cmp(&mcs_ar.f[b])).filter(|_| n > CENTRE_FIRST_MAX_N);
                    let x0 = k.map_or_else(|| (0..n).map(|i| 0.5 * (u[i] + v[i])).collect(), |k| mcs_ar.x[k].clone());
                    (x0, mcs_ar.clone())
                } else if let Some(x) = sep_start.take() {
                    (x, restart_archive(n))
                } else {
                    (restart_point(&mut rng, u, v, &tabu, cfg.tabu), restart_archive(n))
                };
                let w = if first { warm.take() } else { None };
                let sigma = w.as_ref().map_or(sigma_l, |w| w.1);
                let (r, xb) = cma_run(
                    &mut f,
                    &mut done,
                    &x0,
                    sigma,
                    lambda_l,
                    left,
                    &mut rng,
                    &mut ar,
                    w.as_ref().map(|w| w.0.as_slice()),
                    (cfg.warm_cap > 0.0).then_some((cfg.warm_cap, cfg.warm_cond)),
                );
                tabu.push(xb);
                lambda_l *= 2;
                sigma_l = (sigma_l / cfg.sigma_decay).max(SIGMA_FLOOR * cfg.sigma_restart.map_or(sigma0, |r| r * width));
                first = false;
                used_l += r;
                if sep_pending && !done() && used + r < max_evals {
                    // Separable and not solved by CMA-ES: coordinate-wise,
                    // from the best point so far; the next L run starts there.
                    sep_pending = false;
                    let (fx, mut x) = best_seen.borrow().clone();
                    let mut ar = Archive::default();
                    let (_, u_c) = coordinatewise(&mut f, &mut done, &mut x, fx, u, v, max_evals - used - r, &mut ar, FALLBACK_GRID);
                    used_l += u_c;
                    sep_start = Some(x);
                    r + u_c
                } else {
                    r
                }
            }
            'S' => {
                let uu = rng.uniform();
                let ratio = (lambda_l as f64 / lambda0 as f64 / 2.0).max(1.0);
                let lambda = ((lambda0 as f64) * ratio.powf(uu * uu)).floor() as usize;
                let sigma = sigma0 * 10f64.powf(-2.0 * rng.uniform());
                let x0 = restart_point(&mut rng, u, v, &tabu, cfg.tabu);
                let mut ar = restart_archive(n);
                let (r, xb) = cma_run(&mut f, &mut done, &x0, sigma, lambda.max(lambda0), left.min(used_l.max(1)), &mut rng, &mut ar, None, None);
                tabu.push(xb);
                used_s += r;
                r
            }
            'Q' => {
                // Quasi-Newton from a uniform random point: a cheap local
                // search for many-basin functions with smooth basins.
                let x0 = restart_point(&mut rng, u, v, &tabu, cfg.tabu);
                let f0 = f(&x0);
                let mut ar = Archive::default();
                let budget = ((cfg.qn_restart * n as f64) as usize).min(left - 1);
                let r = qn::bfgs(&mut f, &mut done, &x0, f0, u, v, budget, qn_h0, &mut ar);
                tabu.push(r.x);
                used_q += r.used + 1;
                r.used + 1
            }
            'F' => {
                let budget = funnel_budget.min(left);
                funnel_budget *= 2;
                let (r, xb, _) = funnel::race(&mut f, &mut done, u, v, budget, &mut rng);
                tabu.push(xb);
                used_f += r;
                r
            }
            'D' => {
                let budget = de_budget.min(left);
                de_budget *= 2;
                let (r, xb, _) = de::lshade(&mut f, &mut done, u, v, budget, &mut rng);
                tabu.push(xb);
                used_d += r;
                r
            }
            _ => {
                let [wl, ws, wm] = weights;
                let target = (used_l as f64 / wl).max(used_s as f64 / ws) * wm;
                let quota = (target as usize).saturating_sub(used_m).max(10 * n).min(left);
                let r = mcs_co.run(&mut f, quota, max_evals - used, &mut mcs_ar, &mut done);
                used_m += r;
                r
            }
        };
        used += u_now;
        if !was_first && u_now > 0 {
            let r = match choice {
                'L' => 0,
                'S' => 1,
                _ => 2,
            };
            runs[r] += 1;
            if best_seen.borrow().0 < before {
                wins[r] += 1;
                holder = r;
            }
        }
        if u_now == 0 {
            // MCS could not take a step in the room left: stop choosing it.
            // Anything else that cannot run means the budget is spent.
            if choice == 'M' && !mcs_co.finished {
                mcs_co.finished = true;
            } else {
                break;
            }
        }
    }
    used
}
