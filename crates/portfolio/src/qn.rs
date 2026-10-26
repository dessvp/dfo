//! Bound-constrained BFGS with finite-difference gradients: the fast
//! low-budget phase of the portfolio.
//!
//! Quasi-Newton methods with finite differences lead the COCO/BBOB archive
//! between about 30·D and 1000·D evaluations on smooth functions (e.g.
//! SLSQP+lq-CMA-ES, BFGS-ros); Schröder et al. (2022) found that handing
//! over to CMA-ES pays only when CMA-ES inherits the quasi-Newton state.
//! [`bfgs`] therefore returns its inverse Hessian and step lengths along
//! with the best point.
//!
//! Two choices decide its speed on ill-conditioned functions:
//!
//! * The initial inverse Hessian is a deliberately large multiple of the
//!   identity, as in SLSQP (B0 = I), with no Shanno-Phua rescaling. BFGS
//!   corrects eigenvalues of H that are too large in a few backtracked
//!   steps, but those that are too small only very slowly (Nocedal 1992);
//!   rescaling to the steepest curvature made every flat direction of a
//!   condition-1e6 ellipsoid too small (20-D: 182·D evaluations to f = 1
//!   instead of 34·D).
//! * Forward differences carry a bias h·A/2 that stalls the search near
//!   f - f* ≈ 1e-6 on rotated problems, so once forward differences stop
//!   making progress the search switches to central differences.
//!
//! Bounds are handled by projection: coordinates at a bound whose descent
//! direction points outwards are frozen for the step, and the step is
//! shortened to stay inside the box.

use dfo_cmaes::lq::Archive;

/// The quasi-Newton direction with the coordinates in `active` held fixed:
/// `p_F = -(B_FF)^-1 g_F` for `B = H^-1`, i.e. `-(H_FF - H_FA H_AA^-1 H_AF) g_F`,
/// computed as `u = H g_F`, `p = -u + H_{.A} H_AA^-1 u_A`. Returns false if
/// `H_AA` is not positive definite.
fn reduced_direction(h: &[f64], g: &[f64], active: &[usize], p: &mut [f64], gf: &mut [f64], haa: &mut Vec<f64>) -> bool {
    let n = g.len();
    let k = active.len();
    gf.copy_from_slice(g);
    for &i in active {
        gf[i] = 0.0;
    }
    kernels::symv(h, gf, p);
    // t = H_AA^-1 u_A by Cholesky.
    haa.clear();
    haa.extend(active.iter().flat_map(|&i| active.iter().map(move |&j| h[i * n + j])));
    let mut t: Vec<f64> = active.iter().map(|&i| p[i]).collect();
    for j in 0..k {
        let mut d = haa[j * k + j];
        for l in 0..j {
            d -= haa[j * k + l] * haa[j * k + l];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        haa[j * k + j] = d;
        for i in j + 1..k {
            let mut s = haa[i * k + j];
            for l in 0..j {
                s -= haa[i * k + l] * haa[j * k + l];
            }
            haa[i * k + j] = s / d;
        }
    }
    for i in 0..k {
        for l in 0..i {
            t[i] -= haa[i * k + l] * t[l];
        }
        t[i] /= haa[i * k + i];
    }
    for i in (0..k).rev() {
        for l in i + 1..k {
            t[i] -= haa[l * k + i] * t[l];
        }
        t[i] /= haa[i * k + i];
    }
    for (r, pr) in p.iter_mut().enumerate() {
        let corr: f64 = active.iter().zip(&t).map(|(&j, &tj)| h[r * n + j] * tj).sum();
        *pr = corr - *pr;
    }
    for &i in active {
        p[i] = 0.0;
    }
    true
}

/// Result of one quasi-Newton phase.
pub struct QnResult {
    pub x: Vec<f64>,
    pub f: f64,
    /// Inverse Hessian approximation, row-major n x n.
    pub h: Vec<f64>,
    /// Mean length of the last accepted steps (0 if none).
    pub step: f64,
    pub used: usize,
    /// Accepted steps, and how many of them needed backtracking.
    pub iters: usize,
    pub backtracked: usize,
    /// Mean of the actual over the predicted (quadratic-model) decrease of
    /// the accepted steps.
    pub fit: f64,
}

/// Dense kernels of the BFGS iteration on a row-major symmetric n x n
/// matrix. For n <= 40 the matrix fits in L1 and the portable loops beat
/// a BLAS call (see `bench_kernels`).
pub mod kernels {
    /// out = H x.
    #[inline]
    pub fn symv(h: &[f64], x: &[f64], out: &mut [f64]) {
        let n = x.len();
        for (row, o) in h.chunks_exact(n).zip(out.iter_mut()) {
            *o = dot(row, x);
        }
    }

    /// Dot product with four independent accumulators, which the compiler
    /// keeps in two NEON registers.
    #[inline]
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let mut acc = [0.0f64; 4];
        let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
        let (ra, rb) = (ca.remainder(), cb.remainder());
        for (x, y) in ca.zip(cb) {
            for k in 0..4 {
                acc[k] = x[k].mul_add(y[k], acc[k]);
            }
        }
        let mut s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
        for (x, y) in ra.iter().zip(rb) {
            s = x.mul_add(*y, s);
        }
        s
    }

    /// BFGS inverse update H += c·s sᵀ - r·(hq sᵀ + s hqᵀ), one fused pass.
    #[inline]
    pub fn update(h: &mut [f64], s: &[f64], hq: &[f64], c: f64, r: f64) {
        let n = s.len();
        for (i, row) in h.chunks_exact_mut(n).enumerate() {
            let (a, b) = (c * s[i] - r * hq[i], r * s[i]);
            for ((hij, &sj), &qj) in row.iter_mut().zip(s).zip(hq) {
                *hij += a * sj - b * qj;
            }
        }
    }

    /// Accelerate (BLAS) versions of the same kernels, for comparison.
    #[cfg(target_os = "macos")]
    pub mod blas {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_dsymv(order: i32, uplo: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
            fn cblas_dsyr2(order: i32, uplo: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
            fn cblas_dsyr(order: i32, uplo: i32, n: i32, alpha: f64, x: *const f64, incx: i32, a: *mut f64, lda: i32);
        }
        const ROW_MAJOR: i32 = 101;
        const UPPER: i32 = 121;

        pub fn symv(h: &[f64], x: &[f64], out: &mut [f64]) {
            let n = x.len() as i32;
            unsafe { cblas_dsymv(ROW_MAJOR, UPPER, n, 1.0, h.as_ptr(), n, x.as_ptr(), 1, 0.0, out.as_mut_ptr(), 1) };
        }

        /// Updates the upper triangle only (as BLAS does).
        pub fn update(h: &mut [f64], s: &[f64], hq: &[f64], c: f64, r: f64) {
            assert!(h.len() >= s.len() * s.len() && hq.len() >= s.len());
            let n = s.len() as i32;
            // SAFETY: h is n x n, s and hq have n entries (asserted above).
            unsafe {
                cblas_dsyr(ROW_MAJOR, UPPER, n, c, s.as_ptr(), 1, h.as_mut_ptr(), n);
                cblas_dsyr2(ROW_MAJOR, UPPER, n, -r, hq.as_ptr(), 1, s.as_ptr(), 1, h.as_mut_ptr(), n);
            }
        }
    }
}

/// Nanoseconds per call of the portable and (on macOS) BLAS kernels:
/// [symv portable, update portable, symv BLAS, update BLAS].
pub fn bench_kernels(n: usize, reps: usize) -> [f64; 4] {
    use std::hint::black_box;
    use std::time::Instant;
    let mut h: Vec<f64> = (0..n * n).map(|k| ((k * 7919) % 13) as f64 * 0.01).collect();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut y = vec![0.0; n];
    let time = |f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        t.elapsed().as_nanos() as f64 / reps as f64
    };
    let mut out = [0.0; 4];
    out[0] = time(&mut || kernels::symv(black_box(&h), black_box(&x), &mut y));
    out[1] = time(&mut || kernels::update(black_box(&mut h), black_box(&x), black_box(&y), 1e-9, 1e-9));
    #[cfg(target_os = "macos")]
    {
        out[2] = time(&mut || kernels::blas::symv(black_box(&h), black_box(&x), &mut y));
        out[3] = time(&mut || kernels::blas::update(black_box(&mut h), black_box(&x), black_box(&y), 1e-9, 1e-9));
    }
    black_box(&h);
    out
}

/// Minimizes from `x0` (value `f0`) within `[u, v]` using at most `budget`
/// evaluations. The initial inverse Hessian is `h0_scale` times unit
/// curvature on a box of width 10. Line-search evaluations are recorded in
/// `ar`.
#[allow(clippy::too_many_arguments)]
pub fn bfgs<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    x0: &[f64],
    f0: f64,
    u: &[f64],
    v: &[f64],
    budget: usize,
    h0_scale: f64,
    ar: &mut Archive,
) -> QnResult {
    let n = x0.len();
    let wmin = (0..n).map(|i| v[i] - u[i]).fold(f64::INFINITY, f64::min);
    let h0 = (0.1 * wmin).powi(2) * h0_scale;
    let mut used = 0usize;
    // Gradient probes sit 1e-8 apart and would make the surrogate's
    // regression design nearly singular, so only line-search points are
    // recorded.
    let mut eval = |y: &[f64], used: &mut usize, keep: bool| {
        let fy = f(y);
        if keep {
            ar.push(y, fy);
        }
        *used += 1;
        fy
    };
    let mut x = x0.to_vec();
    let mut fx = f0;
    let mut h = vec![0.0; n * n];
    reset(&mut h, n, h0);
    let mut g = vec![0.0; n];
    let mut steps: Vec<f64> = Vec::new();
    let mut y = vec![0.0; n];

    // Finite-difference gradient at (x, fx): forward (n evaluations, step
    // sqrt(eps)) or central (2n, step eps^(1/3)), one-sided at bounds.
    let grad = |x: &[f64], fx: f64, central: bool, g: &mut [f64], y: &mut [f64], used: &mut usize, eval: &mut dyn FnMut(&[f64], &mut usize, bool) -> f64| {
        y.copy_from_slice(x);
        for i in 0..n {
            let scale = x[i].abs().max(1.0);
            if central {
                let hi = (x[i] + 6.0e-6 * scale) - x[i];
                if x[i] - hi >= u[i] && x[i] + hi <= v[i] {
                    y[i] = x[i] + hi;
                    let fp = eval(y, used, false);
                    y[i] = x[i] - hi;
                    let fm = eval(y, used, false);
                    g[i] = (fp - fm) / (2.0 * hi);
                    y[i] = x[i];
                    continue;
                }
            }
            let mut hi = 1.5e-8 * scale;
            if x[i] + hi > v[i] {
                hi = -hi;
            }
            // The difference actually representable at x.
            let hi = (x[i] + hi) - x[i];
            y[i] = x[i] + hi;
            g[i] = (eval(y, used, false) - fx) / hi;
            y[i] = x[i];
        }
    };
    if budget < 2 * n + 2 || done() {
        return QnResult { x, f: fx, h, step: 0.0, used, iters: 0, backtracked: 0, fit: 0.0 };
    }
    let mut central = false;
    grad(&x, fx, central, &mut g, &mut y, &mut used, &mut eval);
    let mut p = vec![0.0; n];
    let mut xn = vec![0.0; n];
    let mut xt = vec![0.0; n];
    let mut gnew = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut hq = vec![0.0; n];
    let mut stalls = 0;
    let (mut iters, mut backtracked, mut fit_sum) = (0usize, 0usize, 0.0f64);
    // A coordinate within a hair of a bound counts as on it: a step
    // shortened to reach a bound may stop just short of it.
    let pushes_out = |x: &[f64], p: &[f64], i: usize| {
        let tol = 1e-12 * (v[i] - u[i]);
        (x[i] <= u[i] + tol && p[i] < 0.0) || (x[i] >= v[i] - tol && p[i] > 0.0)
    };
    let frozen = |x: &[f64], p: &mut [f64]| {
        for i in 0..n {
            if pushes_out(x, p, i) {
                p[i] = 0.0;
            }
        }
    };
    let mut active: Vec<usize> = Vec::new();
    let (mut gf, mut haa) = (vec![0.0; n], Vec::new());
    loop {
        let gcost = if central { 2 * n } else { n };
        if done() || used + gcost + 2 > budget {
            break;
        }
        // Direction: the quasi-Newton step in the coordinates not held at a
        // bound, adding bounds that the step pushes against until none do.
        active.clear();
        kernels::symv(&h, &g, &mut p);
        p.iter_mut().for_each(|t| *t = -*t);
        loop {
            let before = active.len();
            for i in 0..n {
                if !active.contains(&i) && pushes_out(&x, &p, i) {
                    active.push(i);
                }
            }
            if active.len() == before {
                break;
            }
            if !reduced_direction(&h, &g, &active, &mut p, &mut gf, &mut haa) {
                // Rounding made H_AA indefinite: plain projection instead.
                kernels::symv(&h, &g, &mut p);
                p.iter_mut().for_each(|t| *t = -*t);
                frozen(&x, &mut p);
                break;
            }
        }
        let mut slope = kernels::dot(&g, &p);
        if !(slope < 0.0) {
            // Not a descent direction: restart from the initial matrix.
            reset(&mut h, n, h0);
            for i in 0..n {
                p[i] = -h0 * g[i];
            }
            frozen(&x, &mut p);
            slope = kernels::dot(&g, &p);
            if !(slope < 0.0) {
                break;
            }
        }
        // Largest feasible step along p.
        let mut amax: f64 = 1.0;
        for i in 0..n {
            if p[i] > 0.0 {
                amax = amax.min((v[i] - x[i]) / p[i]);
            } else if p[i] < 0.0 {
                amax = amax.min((u[i] - x[i]) / p[i]);
            }
        }
        let mut a = amax.max(0.0);
        if a == 0.0 {
            break;
        }
        // Armijo backtracking with safeguarded quadratic interpolation; the
        // large initial H may overshoot by orders of magnitude, so the
        // contraction goes down to a factor 100 per try.
        let pinf = p.iter().fold(0.0f64, |m, t| m.max(t.abs()));
        let xinf = x.iter().fold(1.0f64, |m, t| m.max(t.abs()));
        let mut fnew;
        let mut tries = 0;
        loop {
            for i in 0..n {
                xn[i] = (x[i] + a * p[i]).clamp(u[i], v[i]);
            }
            fnew = eval(&xn, &mut used, true);
            // Give up once the trial step no longer moves x measurably.
            if fnew <= fx + 1e-4 * a * slope || a * pinf <= 1e-9 * xinf || tries >= 12 || used + gcost + 1 > budget || done() {
                break;
            }
            let denom = 2.0 * (fnew - fx - slope * a);
            let aq = if denom > 0.0 { -slope * a * a / denom } else { 0.5 * a };
            a = aq.clamp(0.01 * a, 0.5 * a);
            tries += 1;
        }
        // A full step cut short at a bound: continue along the projected
        // path (one evaluation per doubling) while f keeps decreasing,
        // instead of spending a gradient per newly active bound.
        if tries == 0 && a == amax && a < 1.0 && fnew < fx {
            while a < 1.0 && used + gcost < budget && !done() {
                let a2 = (2.0 * a).min(1.0);
                for i in 0..n {
                    xt[i] = (x[i] + a2 * p[i]).clamp(u[i], v[i]);
                }
                let ft = eval(&xt, &mut used, true);
                if !(ft < fnew) {
                    break;
                }
                a = a2;
                fnew = ft;
                xn.copy_from_slice(&xt);
            }
        }
        if !(fnew < fx) {
            if central || used + 2 * n + 2 > budget || done() {
                break;
            }
            // Forward differences exhausted: refine the gradient.
            central = true;
            stalls = 0;
            grad(&x, fx, central, &mut g, &mut y, &mut used, &mut eval);
            continue;
        }
        let slen = (0..n).map(|i| (xn[i] - x[i]).powi(2)).sum::<f64>().sqrt();
        steps.push(slen);
        iters += 1;
        backtracked += (tries > 0) as usize;
        // The model m(a) = a·slope + a²/2·(-slope) (BFGS step at a = 1)
        // predicts a decrease of -slope·a·(1 - a/2).
        fit_sum += (fx - fnew) / (-slope * a * (1.0 - 0.5 * a)).max(1e-300);
        // A step cut short by a bound may gain little without the gradient
        // being exhausted.
        let truncated = a == amax && amax < 1.0;
        if fx - fnew <= 1e-13 * fx.abs().max(1e-300) && !truncated {
            stalls += 1;
        } else {
            stalls = 0;
        }
        if used + gcost > budget || done() {
            x.copy_from_slice(&xn);
            fx = fnew;
            break;
        }
        grad(&xn, fnew, central, &mut gnew, &mut y, &mut used, &mut eval);
        for i in 0..n {
            s[i] = xn[i] - x[i];
            q[i] = gnew[i] - g[i];
        }
        let sq = kernels::dot(&s, &q);
        let qq = kernels::dot(&q, &q);
        if sq > 1e-12 * qq.sqrt() * slen {
            kernels::symv(&h, &q, &mut hq);
            let qhq = kernels::dot(&q, &hq);
            let r = 1.0 / sq;
            kernels::update(&mut h, &s, &hq, r * (1.0 + r * qhq), r);
        }
        x.copy_from_slice(&xn);
        fx = fnew;
        g.copy_from_slice(&gnew);
        if slen <= 1e-12 * wmin || stalls >= 2 {
            if central || used + 2 * n + 2 > budget {
                break;
            }
            central = true;
            stalls = 0;
            grad(&x, fx, central, &mut g, &mut y, &mut used, &mut eval);
        }
    }
    let k = steps.len().saturating_sub(3);
    let step = if steps.is_empty() { 0.0 } else { steps[k..].iter().sum::<f64>() / (steps.len() - k) as f64 };
    let fit = if iters > 0 { fit_sum / iters as f64 } else { 0.0 };
    QnResult { x, f: fx, h, step, used, iters, backtracked, fit }
}

fn reset(h: &mut [f64], n: usize, scale: f64) {
    h.iter_mut().for_each(|t| *t = 0.0);
    for i in 0..n {
        h[i * n + i] = scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_rotated_ellipsoid() {
        let n = 10;
        // Ill-conditioned quadratic with a fixed rotation (Householder).
        let w: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let ww: f64 = w.iter().map(|t| t * t).sum();
        let f = |x: &[f64]| {
            let d: Vec<f64> = x.iter().map(|t| t - 1.0).collect();
            let wd: f64 = (0..n).map(|i| w[i] * d[i]).sum();
            (0..n)
                .map(|i| {
                    let z = d[i] - 2.0 * w[i] * wd / ww;
                    1e6f64.powf(i as f64 / (n - 1) as f64) * z * z
                })
                .sum::<f64>()
        };
        let mut ff = f;
        let mut ar = Archive::default();
        let x0 = vec![-2.0; n];
        let f0 = ff(&x0);
        let r = bfgs(&mut ff, &mut || false, &x0, f0, &vec![-5.0; n], &vec![5.0; n], 1000, 1.0, &mut ar);
        assert!(r.f < 1e-8, "f = {} after {} evaluations", r.f, r.used);
    }

    #[test]
    fn reduced_direction_minimizes_on_free_coordinates() {
        // H = inverse of an SPD B; the direction must solve B_FF p_F = -g_F.
        let n = 6;
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] = 1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 };
            }
        }
        // Invert B by Gauss-Jordan.
        let mut m = b.clone();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        for c in 0..n {
            let d = m[c * n + c];
            for j in 0..n {
                m[c * n + j] /= d;
                h[c * n + j] /= d;
            }
            for r in 0..n {
                if r != c {
                    let fct = m[r * n + c];
                    for j in 0..n {
                        m[r * n + j] -= fct * m[c * n + j];
                        h[r * n + j] -= fct * h[c * n + j];
                    }
                }
            }
        }
        let g: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7).sin()).collect();
        let active = [1usize, 4];
        let (mut p, mut gf, mut haa) = (vec![0.0; n], vec![0.0; n], Vec::new());
        assert!(reduced_direction(&h, &g, &active, &mut p, &mut gf, &mut haa));
        for &i in &active {
            assert_eq!(p[i], 0.0);
        }
        for i in (0..n).filter(|i| !active.contains(i)) {
            let r: f64 = (0..n).map(|j| b[i * n + j] * p[j]).sum::<f64>() + g[i];
            assert!(r.abs() < 1e-12, "row {i}: residual {r}");
        }
    }

    #[test]
    fn respects_bounds() {
        let n = 3;
        let mut f = |x: &[f64]| x.iter().map(|t| (t + 10.0).powi(2)).sum::<f64>();
        let mut ar = Archive::default();
        let x0 = vec![0.0; n];
        let f0 = f(&x0);
        let r = bfgs(&mut f, &mut || false, &x0, f0, &vec![-5.0; n], &vec![5.0; n], 400, 1.0, &mut ar);
        assert!(r.x.iter().all(|&t| (-5.0..=5.0).contains(&t)));
        assert!(r.x.iter().all(|&t| (t + 5.0).abs() < 1e-9), "{:?}", r.x);
    }

    #[test]
    fn kernels_match_blas() {
        let n = 13;
        let mut h: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] = 1.0 / (1.0 + (i + j) as f64);
            }
        }
        let x: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let (mut y1, mut y2) = (vec![0.0; n], vec![0.0; n]);
        kernels::symv(&h, &x, &mut y1);
        #[cfg(target_os = "macos")]
        {
            let hq: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            kernels::blas::symv(&h, &x, &mut y2);
            assert!(y1.iter().zip(&y2).all(|(a, b)| (a - b).abs() < 1e-12));
            let mut h2 = h.clone();
            kernels::update(&mut h, &x, &hq, 0.3, 0.7);
            kernels::blas::update(&mut h2, &x, &hq, 0.3, 0.7);
            for i in 0..n {
                for j in i..n {
                    assert!((h[i * n + j] - h2[i * n + j]).abs() < 1e-12);
                }
            }
        }
        let _ = &mut y2;
    }
}
