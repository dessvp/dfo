//! Linear-quadratic global surrogate for CMA-ES, after Hansen, "A Global
//! Surrogate Assisted CMA-ES" (GECCO 2019): each generation the candidates
//! are ranked by a least-squares model fitted to the most recent evaluated
//! points, and evaluated in model order until the model ranks the recent
//! true values well (Kendall tau-b >= 0.85). All candidates are then
//! ranked by their model values, shifted so that the best equals the best
//! true value (pycma's SurrogatePopulation).
//!
//! The model is fitted in the whitened coordinates of the search
//! distribution at fit time, `y = D^-1 B' (x - m) / sigma`, which keeps the
//! least-squares problem well conditioned and makes the diagonal quadratic
//! model (used above `FULL_MAX_N` dimensions) capture the learned rotation.
//! Each model keeps its own whitening, so a full quadratic model, which is
//! invariant to affine changes of coordinates, stays valid after the
//! distribution moves and is reused until new points arrive.
//!
//! The normal equations are formed with a register-tiled `A A'` kernel and
//! solved by a vectorized Cholesky factorization; nothing is allocated per
//! fit once the workspace has grown.
//!
//! The model-building rules follow pycma's `fitness_models` module
//! (BSD-3-Clause, Copyright (c) 2014 Inria; authors N. Hansen, P. Baudis,
//! Y. Akimoto): the number of points per model and the window, the
//! weights, the switch between model kinds, the number of points for
//! Kendall tau, the growth of the evaluated fraction and the shift to the
//! best true value. They are reimplemented here, not translated; see
//! THIRD-PARTY-NOTICES.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cmaes::{Cmaes, Rng, default_lambda};

/// Full quadratic models up to this dimension; diagonal above (settable for
/// ablations).
pub static FULL_MAX_N: AtomicUsize = AtomicUsize::new(40);
const TAU_TRUTH: f64 = 0.85;

/// Evaluated points, in evaluation order, plus the model fitting workspace.
#[derive(Default, Clone)]
pub struct Archive {
    pub x: Vec<Vec<f64>>,
    pub f: Vec<f64>,
    /// Fit at most diagonal quadratic models (cheap in any dimension).
    pub diagonal_only: bool,
    ws: Workspace,
}

impl Archive {
    pub fn push(&mut self, x: &[f64], f: f64) {
        self.x.push(x.to_vec());
        self.f.push(f);
    }

    /// Drops all but the last `keep` points once there are twice as many
    /// (the model only reads a recent window).
    fn trim(&mut self, keep: usize) {
        if self.f.len() > 2 * keep.max(1) {
            let cut = self.f.len() - keep;
            self.x.drain(..cut);
            self.f.drain(..cut);
            self.ws.cached.kind = None;
        }
    }

    /// Forgets all points.
    fn clear(&mut self) {
        self.x.clear();
        self.f.clear();
        self.ws.cached.kind = None;
    }
}

/// Per-run switch that pauses the model while it does not save
/// evaluations, in dimensions where fitting is expensive: a running mean of
/// the fraction of each population evaluated; above `MODEL_OFF` the model
/// is paused for `pause` generations (doubling each time), then tried again.
#[derive(Clone, Debug)]
pub struct Gate {
    smoothed: f64,
    off: usize,
    pause: usize,
}

impl Default for Gate {
    fn default() -> Self {
        Gate { smoothed: 0.0, off: 0, pause: 4 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Linear,
    Diagonal,
    Full,
}

fn n_coef(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Linear => n + 1,
        Kind::Diagonal => 2 * n + 1,
        Kind::Full => (n + 1) * (n + 2) / 2,
    }
}

/// A fitted model: `coef` times the features of `y = w (x - center)`.
#[derive(Default, Clone)]
struct Model {
    kind: Option<Kind>,
    coef: Vec<f64>,
    center: Vec<f64>,
    /// Whitening, row-major n x n: `y = w (x - center)`.
    w: Vec<f64>,
    /// Its inverse: `x = center + inv y`.
    inv: Vec<f64>,
    /// Archive length and window the model was fitted on.
    nf: usize,
    window: usize,
}

/// Scratch space reused across fits.
#[derive(Default, Clone)]
struct Workspace {
    /// Weighted features, one row of p per point, and the weighted targets.
    a: Vec<f64>,
    t: Vec<f64>,
    scratch: Vec<f64>,
    ata: Vec<f64>,
    atb: Vec<f64>,
    gram: Vec<f64>,
    coef: Vec<f64>,
    idx: Vec<usize>,
    y: Vec<f64>,
    pred: Vec<f64>,
    order: Vec<usize>,
    model: Model,
    /// The last model fitted, kept for reuse while the archive is unchanged.
    cached: Model,
}

/// Dense kernels of the fit: Apple's Accelerate (the AMX matrix unit) on
/// macOS for problems large enough to pay for the call, the portable NEON
/// kernels below otherwise.
mod accel {
    #[cfg(target_os = "macos")]
    mod ffi {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            pub fn cblas_dsyrk(order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, beta: f64, c: *mut f64, ldc: i32);
            pub fn cblas_dgemv(
                order: i32,
                trans: i32,
                m: i32,
                n: i32,
                alpha: f64,
                a: *const f64,
                lda: i32,
                x: *const f64,
                incx: i32,
                beta: f64,
                y: *mut f64,
                incy: i32,
            );
            #[link_name = "dpotrf$NEWLAPACK"]
            pub fn dpotrf_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32) -> i32;
            #[link_name = "dpotrs$NEWLAPACK"]
            pub fn dpotrs_(
                uplo: *const u8,
                n: *const i32,
                nrhs: *const i32,
                a: *const f64,
                lda: *const i32,
                b: *mut f64,
                ldb: *const i32,
                info: *mut i32,
            ) -> i32;
        }
        pub const ROW_MAJOR: i32 = 101;
        pub const NO_TRANS: i32 = 111;
        pub const LOWER: i32 = 122;
    }

    /// Below this many coefficients the call overhead outweighs the AMX.
    #[cfg(target_os = "macos")]
    const MIN_P: usize = 48;

    /// Lower triangle (row-major p x p) of `A A'`, `A` being p rows of length m.
    pub fn syrk_lower(a: &[f64], p: usize, m: usize, out: &mut [f64]) {
        #[cfg(target_os = "macos")]
        if p >= MIN_P && m > 0 {
            crate::cmaes::blas_single_thread();
            // SAFETY: a is p x m row-major, out is p x p; sizes checked by the slices' lengths.
            assert!(a.len() >= p * m && out.len() >= p * p);
            unsafe {
                ffi::cblas_dsyrk(ffi::ROW_MAJOR, ffi::LOWER, ffi::NO_TRANS, p as i32, m as i32, 1.0, a.as_ptr(), m as i32, 0.0, out.as_mut_ptr(), p as i32)
            };
            return;
        }
        super::syrk_lower(a, p, m, out)
    }

    /// `A t` for `A` stored as p rows of length m.
    pub fn gemv(a: &[f64], t: &[f64], p: usize, m: usize, out: &mut [f64]) {
        assert!(a.len() >= p * m && t.len() >= m && out.len() >= p);
        #[cfg(target_os = "macos")]
        if p >= MIN_P && m > 0 {
            crate::cmaes::blas_single_thread();
            // SAFETY: sizes asserted above.
            unsafe { ffi::cblas_dgemv(ffi::ROW_MAJOR, ffi::NO_TRANS, p as i32, m as i32, 1.0, a.as_ptr(), m as i32, t.as_ptr(), 1, 0.0, out.as_mut_ptr(), 1) };
            return;
        }
        for i in 0..p {
            out[i] = super::dot(&a[i * m..(i + 1) * m], &t[..m]);
        }
    }

    /// Solves `a x = b` (row-major, lower triangle of `a` read) by Cholesky.
    pub fn cholesky_solve(a: &mut [f64], b: &mut [f64], p: usize) -> bool {
        #[cfg(target_os = "macos")]
        if p >= MIN_P {
            crate::cmaes::blas_single_thread();
            // Row-major lower = column-major upper.
            let (n, one, mut info) = (p as i32, 1i32, 0i32);
            assert!(a.len() >= p * p && b.len() >= p);
            unsafe { ffi::dpotrf_(b"U".as_ptr(), &n, a.as_mut_ptr(), &n, &mut info) };
            if info != 0 {
                return false;
            }
            unsafe { ffi::dpotrs_(b"U".as_ptr(), &n, &one, a.as_ptr(), &n, b.as_mut_ptr(), &n, &mut info) };
            return info == 0 && b[..p].iter().all(|t| t.is_finite());
        }
        super::cholesky_solve(a, b, p)
    }
}

/// `a . b` with eight independent accumulators.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_f64(0.0); 4];
        let chunks = n / 8;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        for c in 0..chunks {
            let o = c * 8;
            for l in 0..4 {
                acc[l] = vfmaq_f64(acc[l], vld1q_f64(pa.add(o + 2 * l)), vld1q_f64(pb.add(o + 2 * l)));
            }
        }
        let s = vaddq_f64(vaddq_f64(acc[0], acc[1]), vaddq_f64(acc[2], acc[3]));
        let mut s = vaddvq_f64(s);
        for k in chunks * 8..n {
            s = a[k].mul_add(b[k], s);
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = [0.0; 8];
        let chunks = n / 8;
        for c in 0..chunks {
            for l in 0..8 {
                acc[l] = a[c * 8 + l].mul_add(b[c * 8 + l], acc[l]);
            }
        }
        let mut s: f64 = acc.iter().sum();
        for k in chunks * 8..n {
            s = a[k].mul_add(b[k], s);
        }
        s
    }
}

/// Lower triangle (row-major, p x p) of `A A'` for `A` stored as `p` rows
/// of length `m`: 4 x 4 register tiles over the rows.
fn syrk_lower(a: &[f64], p: usize, m: usize, out: &mut [f64]) {
    let row = |i: usize| &a[i * m..(i + 1) * m];
    let mut i = 0;
    while i < p {
        let ib = (p - i).min(4);
        let mut j = 0;
        while j <= i {
            let jb = (p - j).min(4);
            if ib == 4 && jb == 4 {
                let t = tile4([row(i), row(i + 1), row(i + 2), row(i + 3)], [row(j), row(j + 1), row(j + 2), row(j + 3)]);
                for (r, tr) in t.iter().enumerate() {
                    for (c, &v) in tr.iter().enumerate() {
                        if j + c <= i + r {
                            out[(i + r) * p + j + c] = v;
                        }
                    }
                }
            } else {
                for r in i..i + ib {
                    for c in j..(j + jb).min(r + 1) {
                        out[r * p + c] = dot(row(r), row(c));
                    }
                }
            }
            j += 4;
        }
        i += 4;
    }
}

/// The 4 x 4 block of dot products between rows `ar` and rows `br`.
#[inline]
fn tile4(ar: [&[f64]; 4], br: [&[f64]; 4]) -> [[f64; 4]; 4] {
    let m = ar[0].len();
    // The NEON loop below reads m entries of every row.
    assert!(ar.iter().chain(&br).all(|r| r.len() == m));
    let mut out = [[0.0; 4]; 4];
    #[cfg(target_arch = "aarch64")]
    let done = unsafe {
        use std::arch::aarch64::*;
        let mut acc = [[vdupq_n_f64(0.0); 4]; 4];
        let pairs = m / 2;
        for k in 0..pairs {
            let o = 2 * k;
            let va = ar.map(|r| vld1q_f64(r.as_ptr().add(o)));
            let vb = br.map(|r| vld1q_f64(r.as_ptr().add(o)));
            for r in 0..4 {
                for c in 0..4 {
                    acc[r][c] = vfmaq_f64(acc[r][c], va[r], vb[c]);
                }
            }
        }
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = vaddvq_f64(acc[r][c]);
            }
        }
        pairs * 2
    };
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    for k in done..m {
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = ar[r][k].mul_add(br[c][k], out[r][c]);
            }
        }
    }
    out
}

/// Solves `a x = b` for symmetric positive definite `a` (row-major, only
/// the lower triangle is read) by Cholesky; returns false if `a` is not
/// positive definite.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], p: usize) -> bool {
    for j in 0..p {
        let (head, tail) = a.split_at_mut(j * p);
        let rj = &mut tail[..p];
        // Row j left of the diagonal: L_jk = (a_jk - L_j,<k . L_k,<k) / L_kk.
        for k in 0..j {
            let rk = &head[k * p..k * p + k];
            rj[k] = (rj[k] - dot(&rj[..k], rk)) / head[k * p + k];
        }
        let d = rj[j] - dot(&rj[..j], &rj[..j]);
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        rj[j] = d.sqrt();
    }
    for i in 0..p {
        b[i] = (b[i] - dot(&a[i * p..i * p + i], &b[..i])) / a[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= a[k * p + i] * b[k];
        }
        b[i] = s / a[i * p + i];
    }
    true
}

/// `y = w (x - c)` for row-major n x n `w`; `d` is scratch.
#[inline]
fn apply_whiten(w: &[f64], c: &[f64], x: &[f64], d: &mut [f64], y: &mut [f64]) {
    let n = c.len();
    for i in 0..n {
        d[i] = x[i] - c[i];
    }
    for j in 0..n {
        y[j] = dot(&w[j * n..(j + 1) * n], &d[..n]);
    }
}

/// The features of `y` into `out` (length `n_coef(kind, n)`).
#[inline]
fn features_into(kind: Kind, y: &[f64], out: &mut [f64]) {
    let n = y.len();
    out[0] = 1.0;
    out[1..=n].copy_from_slice(y);
    match kind {
        Kind::Linear => {}
        Kind::Diagonal => {
            for i in 0..n {
                out[n + 1 + i] = y[i] * y[i];
            }
        }
        Kind::Full => {
            let mut k = n + 1;
            for i in 0..n {
                let yi = y[i];
                for (o, &yj) in out[k..k + n - i].iter_mut().zip(&y[i..]) {
                    *o = yi * yj;
                }
                k += n - i;
            }
        }
    }
}

impl Model {
    /// Sets the whitening to the distribution of `es`.
    fn set_frame(&mut self, es: &Cmaes) {
        let n = es.n;
        let (b, d) = (es.b_mat(), es.d_vec());
        self.center.clear();
        self.center.extend_from_slice(&es.mean);
        self.w.resize(n * n, 0.0);
        self.inv.resize(n * n, 0.0);
        for i in 0..n {
            for j in 0..n {
                self.w[j * n + i] = b[i * n + j] / (es.sigma * d[j]);
                self.inv[i * n + j] = es.sigma * b[i * n + j] * d[j];
            }
        }
    }

    fn predict(&self, x: &[f64], scratch: &mut Vec<f64>) -> f64 {
        let n = self.center.len();
        scratch.resize(2 * n, 0.0);
        let (y, d) = scratch.split_at_mut(n);
        apply_whiten(&self.w, &self.center, x, d, y);
        let c = &self.coef;
        let mut s = c[0] + dot(&c[1..=n], y);
        match self.kind {
            Some(Kind::Diagonal) => {
                for i in 0..n {
                    s += c[n + 1 + i] * y[i] * y[i];
                }
            }
            Some(Kind::Full) => {
                let mut k = n + 1;
                for i in 0..n {
                    let t = dot(&c[k..k + n - i], &y[i..]);
                    k += n - i;
                    s += y[i] * t;
                }
            }
            _ => {}
        }
        s
    }

    /// Minimizer of a quadratic model, in the original coordinates, if the
    /// model is convex.
    fn xopt(&self) -> Option<Vec<f64>> {
        let n = self.center.len();
        let g = &self.coef[1..=n];
        let yopt: Vec<f64> = match self.kind? {
            Kind::Linear => return None,
            Kind::Diagonal => {
                let a = &self.coef[n + 1..];
                if a.iter().any(|&t| t <= 0.0) {
                    return None;
                }
                (0..n).map(|i| -g[i] / (2.0 * a[i])).collect()
            }
            Kind::Full => {
                // f = c + g'y + sum_{i<=j} h_ij y_i y_j; gradient g + H2 y with
                // H2_ii = 2 h_ii and H2_ij = h_ij (lower triangle suffices).
                let mut h2 = vec![0.0; n * n];
                let mut k = n + 1;
                for i in 0..n {
                    for j in i..n {
                        let c = self.coef[k];
                        k += 1;
                        h2[j * n + i] = if i == j { 2.0 * c } else { c };
                    }
                }
                let mut rhs: Vec<f64> = g.iter().map(|t| -t).collect();
                if !cholesky_solve(&mut h2, &mut rhs, n) {
                    return None;
                }
                rhs
            }
        };
        Some((0..n).map(|i| self.center[i] + dot(&self.inv[i * n..(i + 1) * n], &yopt)).collect())
    }
}

/// Fits the richest model the archive supports into `ar.ws.model`, by
/// least squares weighted by rank (20 for the best point down to 1) on the
/// best `TRUNCATION` part of the most recent points (pycma's LQModel
/// defaults). A full model fitted on the same points is reused as is.
/// Returns whether a model is available.
fn fit(es: &Cmaes, ar: &mut Archive) -> bool {
    const TRUNCATION: f64 = 0.75;
    const MIN_RELATIVE: f64 = 1.1;
    let n = es.n;
    let kinds: &[Kind] = if richest(n, ar) == Kind::Full { &[Kind::Full, Kind::Diagonal, Kind::Linear] } else { &[Kind::Diagonal, Kind::Linear] };
    // Archive window: the most recent max(lambda, 2 * df_max) points.
    let max_df = n_coef(kinds[0], n);
    let nf = ar.f.len();
    let window = nf.min(es.lambda.max(2 * max_df));
    let ws = &mut ar.ws;
    let Some(&kind) = kinds.iter().find(|&&k| window as f64 * TRUNCATION >= MIN_RELATIVE * n_coef(k, n) as f64) else {
        ws.model.kind = None;
        return false;
    };
    if kind == Kind::Full && ws.cached.kind == Some(Kind::Full) && ws.cached.nf == nf && ws.cached.window == window {
        ws.model.clone_from(&ws.cached);
        return true;
    }
    let p = n_coef(kind, n);
    let start = nf - window;
    ws.idx.clear();
    ws.idx.extend(start..nf);
    let f = &ar.f;
    ws.idx.sort_by(|&a, &b| f[a].total_cmp(&f[b]));
    let m = ((TRUNCATION * (window + 1) as f64) as usize).max(p + 2).min(window);
    let model = &mut ws.model;
    model.set_frame(es);
    // Targets relative to the best value: the constant coefficient absorbs
    // it, and the others do not depend on how far f is from zero.
    let fref = f[ws.idx[0]];
    // Weighted features, p rows of m (the layout the AMX syrk runs fastest
    // on), built BLOCK points at a time so the transposed writes stay
    // contiguous.
    const BLOCK: usize = 8;
    ws.a.resize(p * m, 0.0);
    ws.t.resize(m, 0.0);
    ws.atb.resize(p, 0.0);
    ws.y.resize(2 * n, 0.0);
    ws.scratch.resize(BLOCK * p, 0.0);
    for r0 in (0..m).step_by(BLOCK) {
        let nb = BLOCK.min(m - r0);
        for b in 0..nb {
            let r = r0 + b;
            let k = ws.idx[r];
            let w = if m > 1 { 20.0 - 19.0 * r as f64 / (m - 1) as f64 } else { 1.0 };
            let (y, d) = ws.y.split_at_mut(n);
            apply_whiten(&model.w, &model.center, &ar.x[k], d, y);
            let row = &mut ws.scratch[b * p..(b + 1) * p];
            features_into(kind, y, row);
            row.iter_mut().for_each(|v| *v *= w);
            ws.t[r] = w * (f[k] - fref);
        }
        for i in 0..p {
            let dst = &mut ws.a[i * m + r0..i * m + r0 + nb];
            for (b, o) in dst.iter_mut().enumerate() {
                *o = ws.scratch[b * p + i];
            }
        }
    }
    // A tiny ridge (1e-14 of the mean diagonal) only guards against an
    // exactly singular system: a larger one biases the fit enough to slow
    // lq-CMA-ES about threefold on ill-conditioned quadratics. If the
    // factorization still fails, retry with a larger one.
    accel::gemv(&ws.a, &ws.t, p, m, &mut ws.atb);
    let mut solved = false;
    ws.ata.resize(p * p, 0.0);
    accel::syrk_lower(&ws.a, p, m, &mut ws.ata);
    let tr: f64 = (0..p).map(|i| ws.ata[i * p + i]).sum();
    for ridge in [1e-14, 1e-10] {
        ws.gram.clone_from(&ws.ata);
        ws.coef.clone_from(&ws.atb);
        for i in 0..p {
            ws.gram[i * p + i] += ridge * tr / p as f64;
        }
        if accel::cholesky_solve(&mut ws.gram, &mut ws.coef, p) {
            solved = true;
            break;
        }
    }
    if !solved {
        model.kind = None;
        ws.cached.kind = None;
        return false;
    }
    ws.coef[0] += fref;
    model.kind = Some(kind);
    model.coef.clone_from(&ws.coef);
    model.nf = nf;
    model.window = window;
    ws.cached.clone_from(model);
    true
}

/// The richest model kind used in dimension `n` with archive `ar`.
fn richest(n: usize, ar: &Archive) -> Kind {
    if n <= FULL_MAX_N.load(Ordering::Relaxed) && !ar.diagonal_only { Kind::Full } else { Kind::Diagonal }
}

/// Kendall's tau-b between two equally long sequences (ties lower it, as
/// in scipy's `kendalltau`, which pycma uses).
fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut conc, mut disc, mut tie_a, mut tie_b) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let (da, db) = (a[i] - a[j], b[i] - b[j]);
            if da == 0.0 {
                tie_a += 1;
            }
            if db == 0.0 {
                tie_b += 1;
            }
            let s = da.signum() * db.signum();
            if da != 0.0 && db != 0.0 {
                if s > 0.0 {
                    conc += 1;
                } else {
                    disc += 1;
                }
            }
        }
    }
    let n0 = (n * n.saturating_sub(1) / 2) as i64;
    let den = (((n0 - tie_a) * (n0 - tie_b)) as f64).sqrt();
    if den > 0.0 { (conc - disc) as f64 / den } else { 0.0 }
}

/// Model values of the population of `es` into `ws.pred`.
fn predict_pop(es: &Cmaes, ws: &mut Workspace) {
    let n = es.n;
    ws.pred.clear();
    for k in 0..es.lambda {
        let v = ws.model.predict(&es.pop[k * n..(k + 1) * n], &mut ws.y);
        ws.pred.push(v);
    }
}

/// One lq-CMA-ES generation (pycma's SurrogatePopulation with model-optimum
/// injection): asks `es` for candidates, injects the model optimum,
/// evaluates candidates in model order (1, then x1.5 more) until Kendall
/// tau on recent points reaches 0.85, and tells `es` the model values
/// shifted to the best true value. Returns evaluations used and whether
/// the run should stop.
///
/// `gate` is per-run state (see `Gate`): while the model is paused the
/// generation is plain CMA-ES, whose points still enter the archive.
pub fn generation<F: FnMut(&[f64]) -> f64>(es: &mut Cmaes, ar: &mut Archive, f: &mut F, rng: &mut Rng, eval_budget: usize, gate: &mut Gate) -> (usize, bool) {
    const MODEL_OFF: f64 = 0.85;
    const MODEL_OFF_MIN_N: usize = 15;
    const MAX_PAUSE: usize = 256;
    let n = es.n;
    let lambda = es.lambda;
    ar.trim(lambda.max(2 * n_coef(richest(n, ar), n)));
    es.ask(rng);
    if gate.off > 0 {
        gate.off -= 1;
        if gate.off == 0 {
            // Try the model again, ready to pause after one poor generation.
            gate.smoothed = MODEL_OFF - 0.05;
        }
        let mut fit_vals = Vec::with_capacity(lambda);
        for k in 0..lambda {
            if k >= eval_budget {
                return (k, true);
            }
            let v = f(&es.pop[k * n..(k + 1) * n]);
            ar.push(&es.pop[k * n..(k + 1) * n], v);
            fit_vals.push(v);
        }
        return (lambda, es.tell(&fit_vals).is_some());
    }
    let mut have = fit(es, ar);
    if have && let Some(xo) = ar.ws.model.xopt() {
        es.inject(lambda - 1, &xo);
    }
    let mut truth: Vec<Option<f64>> = vec![None; lambda];
    let mut used = 0;
    let mut number = 1 + ((lambda as f64 * 0.02) as usize).max((3.0 / 0.75) as usize - ar.f.len().min(4));
    loop {
        let mut order = std::mem::take(&mut ar.ws.order);
        order.clear();
        order.extend(0..lambda);
        if have {
            predict_pop(es, &mut ar.ws);
            let pred = &ar.ws.pred;
            order.sort_by(|&a, &b| pred[a].total_cmp(&pred[b]));
        }
        let mut evaluated = truth.iter().filter(|t| t.is_some()).count();
        for &k in &order {
            if evaluated >= number {
                break;
            }
            if truth[k].is_none() {
                if used >= eval_budget {
                    ar.ws.order = order;
                    return (used, true);
                }
                let v = f(&es.pop[k * n..(k + 1) * n]);
                truth[k] = Some(v);
                ar.push(&es.pop[k * n..(k + 1) * n], v);
                used += 1;
                evaluated += 1;
            }
        }
        ar.ws.order = order;
        have = fit(es, ar);
        if evaluated >= lambda {
            break;
        }
        if !have {
            number += number.div_ceil(2);
            continue;
        }
        let nt = (15.0f64).max((1.2 * evaluated as f64).min(0.75 * lambda as f64)) as usize;
        let nt = nt.min(ar.f.len());
        if nt >= 3 {
            let nf = ar.f.len();
            let ws = &mut ar.ws;
            ws.pred.clear();
            for k in nf - nt..nf {
                let v = ws.model.predict(&ar.x[k], &mut ws.y);
                ws.pred.push(v);
            }
            if kendall_tau(&ws.pred, &ar.f[nf - nt..]) >= TAU_TRUTH {
                break;
            }
        }
        number += number.div_ceil(2);
    }
    let all = truth.iter().all(|t| t.is_some());
    let mut x = vec![0.0; n];
    for (k, t) in truth.iter().enumerate() {
        if let Some(v) = *t {
            x.copy_from_slice(&es.pop[k * n..(k + 1) * n]);
            es.observe(&x, v);
        }
    }
    let fit_vals: Vec<f64> = if all || !have {
        truth.iter().map(|t| t.unwrap_or(f64::INFINITY)).collect()
    } else {
        predict_pop(es, &mut ar.ws);
        let pred = &ar.ws.pred;
        let fmin_true = truth.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        let pmin = pred.iter().copied().fold(f64::INFINITY, f64::min);
        pred.iter().map(|p| p - pmin + fmin_true).collect()
    };
    // Only where fitting is expensive: below MODEL_OFF_MIN_N the model costs
    // microseconds, and its ordering and optimum injection still help.
    if n >= MODEL_OFF_MIN_N && have && ar.ws.model.kind == Some(richest(n, ar)) {
        gate.smoothed = 0.9 * gate.smoothed + 0.1 * used as f64 / lambda as f64;
        if gate.smoothed > MODEL_OFF {
            gate.off = gate.pause;
            gate.pause = (2 * gate.pause).min(MAX_PAUSE);
        }
    }
    (used, es.tell_ranks(&fit_vals).is_some())
}

/// IPOP lq-CMA-ES: restarts from uniform points in `[lo, hi]` with doubling
/// population until `done()` or `max_evals`. `ar` may hold earlier
/// evaluations (e.g. from MCS) to warm-start the model. Returns the
/// evaluations used.
///
/// `first`, if given, is the mean and sigma of the first run (e.g. the best
/// point MCS found).
#[allow(clippy::too_many_arguments)]
pub fn ipop_lq<F: FnMut(&[f64]) -> f64, D: FnMut() -> bool>(
    f: &mut F,
    mut done: D,
    lo: &[f64],
    hi: &[f64],
    max_evals: usize,
    seed: u64,
    ar: &mut Archive,
    first: Option<(&[f64], f64)>,
) -> usize {
    let n = lo.len();
    let mut rng = Rng::new(seed);
    let mut used = 0;
    let mut lambda = default_lambda(n);
    let mut first = first;
    while used < max_evals && !done() {
        let (x0, sigma): (Vec<f64>, f64) = match first.take() {
            Some((x, s)) => (x.to_vec(), s),
            None => ((0..n).map(|i| lo[i] + rng.uniform() * (hi[i] - lo[i])).collect(), 0.2 * (0..n).map(|i| hi[i] - lo[i]).fold(f64::INFINITY, f64::min)),
        };
        let mut es = Cmaes::new(&x0, sigma, lambda);
        let mut gate = Gate::default();
        loop {
            if used >= max_evals || done() {
                return used;
            }
            let (u, stop) = generation(&mut es, ar, f, &mut rng, max_evals - used, &mut gate);
            used += u;
            if stop {
                break;
            }
        }
        // A fresh model per restart (pycma's default): points of the old
        // basin would hold the best ranks and pull the new run back.
        ar.clear();
        lambda *= 2;
    }
    used
}

/// Times the fit kernels (portable NEON vs Accelerate) at size p x m;
/// for benchmarks only.
#[doc(hidden)]
pub fn bench_kernels(p: usize, m: usize, reps: usize) -> [f64; 4] {
    let a: Vec<f64> = (0..p * m).map(|k| ((k * 37 % 101) as f64 - 50.0) / 7.0).collect();
    let mut o = vec![0.0; p * p];
    let mut t = [f64::INFINITY; 4];
    for _ in 0..reps {
        let s = std::time::Instant::now();
        syrk_lower(&a, p, m, &mut o);
        t[0] = t[0].min(s.elapsed().as_secs_f64());
        let s = std::time::Instant::now();
        accel::syrk_lower(&a, p, m, &mut o);
        t[1] = t[1].min(s.elapsed().as_secs_f64());
        for i in 0..p {
            o[i * p + i] += 1.0;
        }
        let (mut c1, mut c2) = (o.clone(), o.clone());
        let (mut b1, mut b2) = (vec![1.0; p], vec![1.0; p]);
        let s = std::time::Instant::now();
        cholesky_solve(&mut c1, &mut b1, p);
        t[2] = t[2].min(s.elapsed().as_secs_f64());
        let s = std::time::Instant::now();
        accel::cholesky_solve(&mut c2, &mut b2, p);
        t[3] = t[3].min(s.elapsed().as_secs_f64());
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syrk_matches_naive() {
        for (p, m) in [(1, 1), (5, 3), (7, 13), (21, 40), (66, 101)] {
            let a: Vec<f64> = (0..p * m).map(|k| ((k * 37 % 101) as f64 - 50.0) / 7.0).collect();
            let mut out = vec![0.0; p * p];
            syrk_lower(&a, p, m, &mut out);
            for i in 0..p {
                for j in 0..=i {
                    let s: f64 = (0..m).map(|k| a[i * m + k] * a[j * m + k]).sum();
                    assert!((out[i * p + j] - s).abs() <= 1e-9 * (1.0 + s.abs()), "p={p} m={m} ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn accelerate_matches_portable() {
        for (p, m) in [(48, 60), (66, 101), (231, 350)] {
            let a: Vec<f64> = (0..p * m).map(|k| ((k * 37 % 101) as f64 - 50.0) / 7.0).collect();
            let (mut o1, mut o2) = (vec![0.0; p * p], vec![0.0; p * p]);
            syrk_lower(&a, p, m, &mut o1);
            accel::syrk_lower(&a, p, m, &mut o2);
            for i in 0..p {
                for j in 0..=i {
                    assert!((o1[i * p + j] - o2[i * p + j]).abs() <= 1e-9 * (1.0 + o1[i * p + j].abs()));
                }
            }
            for i in 0..p {
                o1[i * p + i] += 1.0;
                o2[i * p + i] += 1.0;
            }
            let mut b1: Vec<f64> = (0..p).map(|i| i as f64 - 3.0).collect();
            let mut b2 = b1.clone();
            assert_eq!(cholesky_solve(&mut o1, &mut b1, p), accel::cholesky_solve(&mut o2, &mut b2, p));
            for i in 0..p {
                assert!((b1[i] - b2[i]).abs() <= 1e-6 * (1.0 + b1[i].abs()), "{i}: {} {}", b1[i], b2[i]);
            }
        }
    }

    /// Rotated ellipsoid (condition 1e6) with a large offset.
    fn ellipsoid(n: usize) -> impl FnMut(&[f64]) -> f64 {
        let mut rng = Rng::new(5);
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
        move |x: &[f64]| {
            1000.0
                + (0..n)
                    .map(|i| {
                        let z: f64 = (0..n).map(|j| r[i * n + j] * (x[j] - 1.0)).sum();
                        1e6f64.powf(i as f64 / (n - 1) as f64) * z * z
                    })
                    .sum::<f64>()
        }
    }

    #[test]
    fn lq_solves_ellipsoid_fast() {
        // pycma's lq-CMA-ES needs about 100 evaluations here (10-D).
        let n = 10;
        let mut f = ellipsoid(n);
        let best = std::cell::Cell::new(f64::INFINITY);
        let mut g = |x: &[f64]| {
            let v = f(x);
            best.set(best.get().min(v));
            v
        };
        let mut ar = Archive::default();
        let used = ipop_lq(&mut g, || best.get() - 1000.0 < 1e-8, &[-5.0; 10], &[5.0; 10], 2000, 3, &mut ar, None);
        assert!(best.get() - 1000.0 < 1e-8 && used < 200, "used {used}, f-fopt {:.2e}", best.get() - 1000.0);
    }

    #[test]
    fn kendall_tau_b_with_ties() {
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-15);
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-15);
        // One tie in b: tau-b = (2 - 0) / sqrt(3 * 2).
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0]) - 2.0 / 6f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn cholesky_solves() {
        let p = 23;
        let b: Vec<f64> = (0..p * p).map(|k| ((k * 53 % 97) as f64 - 48.0) / 9.0).collect();
        let mut a = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                a[i * p + j] = (0..p).map(|k| b[i * p + k] * b[j * p + k]).sum::<f64>() + if i == j { 1.0 } else { 0.0 };
            }
        }
        let x: Vec<f64> = (0..p).map(|i| i as f64 - 7.0).collect();
        let mut rhs: Vec<f64> = (0..p).map(|i| (0..p).map(|j| a[i * p + j] * x[j]).sum()).collect();
        let mut l = a.clone();
        assert!(cholesky_solve(&mut l, &mut rhs, p));
        for i in 0..p {
            assert!((rhs[i] - x[i]).abs() < 1e-8, "{i}: {} vs {}", rhs[i], x[i]);
        }
    }
}
