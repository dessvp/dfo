//! CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016) with
//! default parameters, as an ask/tell state machine, plus IPOP restarts
//! (Auger & Hansen 2005). Used by the hybrid engine; benchmarked alone to
//! check it against the archived CMA-ES results.

/// xorshift64* with Box-Muller normals.
#[derive(Clone)]
pub struct Rng {
    s: u64,
    spare: Option<f64>,
}

impl Rng {
    /// Seeds through splitmix64, so that neighbouring seeds give unrelated
    /// streams.
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng { s: (z ^ (z >> 31)).max(1), spare: None }
    }

    pub fn uniform(&mut self) -> f64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        ((self.s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

/// Eigendecomposition of a symmetric n×n matrix (row-major): eigenvalues
/// into `d` and eigenvectors as columns of `b` (row-major). LAPACK's
/// divide-and-conquer `dsyevd` on macOS (Accelerate), cyclic Jacobi
/// elsewhere or if it fails.
pub fn sym_eig(a: &[f64], n: usize, d: &mut [f64], b: &mut [f64]) {
    #[cfg(target_os = "macos")]
    if lapack_eig(a, n, d, b) {
        return;
    }
    jacobi_eig(a, n, d, b)
}

/// Keeps Accelerate's BLAS and LAPACK on the calling thread: the benchmark
/// runs one process per core, and on one thread the AMX kernels are also
/// faster (40-D model fit: A'A 7.1 ms instead of 10.2 ms). Setting
/// `VECLIB_MAXIMUM_THREADS` from inside the process comes too late.
#[cfg(target_os = "macos")]
pub fn blas_single_thread() {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn BLASSetThreading(threading: u32) -> i32;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: plain FFI call with a valid enum value (1 = single threaded).
    ONCE.call_once(|| unsafe {
        BLASSetThreading(1);
    });
}

#[cfg(target_os = "macos")]
fn lapack_eig(a: &[f64], n: usize, d: &mut [f64], b: &mut [f64]) -> bool {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[link_name = "dsyevd$NEWLAPACK"]
        fn dsyevd_(
            jobz: *const u8,
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            w: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            iwork: *mut i32,
            liwork: *const i32,
            info: *mut i32,
        ) -> i32;
    }
    thread_local! {
        static WORK: std::cell::RefCell<(Vec<f64>, Vec<f64>, Vec<i32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    }
    if n == 0 || a.len() < n * n || b.len() < n * n || d.len() < n {
        return false;
    }
    blas_single_thread();
    WORK.with(|w| {
        let (m, work, iwork) = &mut *w.borrow_mut();
        // Symmetric, so row-major is column-major; the eigenvectors come back
        // as the columns of the column-major result, i.e. its rows here.
        m.clear();
        m.extend_from_slice(&a[..n * n]);
        let (ni, lwork, liwork, mut info) = (n as i32, (1 + 6 * n + 2 * n * n) as i32, (3 + 5 * n) as i32, 0i32);
        work.resize(lwork as usize, 0.0);
        iwork.resize(liwork as usize, 0);
        // SAFETY: all buffers are sized as dsyevd requires for jobz = 'V'.
        unsafe {
            dsyevd_(b"V".as_ptr(), b"U".as_ptr(), &ni, m.as_mut_ptr(), &ni, d.as_mut_ptr(), work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &liwork, &mut info)
        };
        if info != 0 || !d[..n].iter().all(|t| t.is_finite()) {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] = m[j * n + i];
            }
        }
        true
    })
}

/// Cyclic Jacobi, until the off-diagonal part is negligible relative to
/// the diagonal.
fn jacobi_eig(a: &[f64], n: usize, d: &mut [f64], b: &mut [f64]) {
    let mut m = a.to_vec();
    b.fill(0.0);
    for i in 0..n {
        b[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        let diag: f64 = (0..n).map(|i| m[i * n + i] * m[i * n + i]).sum();
        if off < 1e-300 || off < 1e-32 * diag {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (bkp, bkq) = (b[k * n + p], b[k * n + q]);
                    b[k * n + p] = c * bkp - s * bkq;
                    b[k * n + q] = s * bkp + c * bkq;
                }
            }
        }
    }
    for i in 0..n {
        d[i] = m[i * n + i];
    }
}

/// Why a CMA-ES run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    TolFun,
    TolX,
    Condition,
    NoEffect,
    Stagnation,
    MaxIter,
}

/// One CMA-ES run (fixed population size).
pub struct Cmaes {
    pub n: usize,
    pub lambda: usize,
    mu: usize,
    w: Vec<f64>,
    /// All lambda recombination weights: the positive w above, then the
    /// negative weights of active CMA (tutorial 2016, eq. 50-53).
    w_all: Vec<f64>,
    mueff: f64,
    cs: f64,
    ds: f64,
    cc: f64,
    c1: f64,
    cmu: f64,
    chin: f64,
    pub mean: Vec<f64>,
    pub sigma: f64,
    sigma0: f64,
    c: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
    ps: Vec<f64>,
    pc: Vec<f64>,
    eigen_at: usize,
    pub iter: usize,
    /// Offspring of the current iteration (row k = candidate k) and their
    /// standard-normal pre-images.
    pub pop: Vec<f64>,
    z: Vec<f64>,
    best_hist: Vec<f64>,
    median_hist: Vec<f64>,
    pub best_f: f64,
    pub best_x: Vec<f64>,
}

impl Cmaes {
    pub fn new(mean: &[f64], sigma: f64, lambda: usize) -> Self {
        let n = mean.len();
        let nf = n as f64;
        let mu = lambda / 2;
        let raw: Vec<f64> = (0..mu).map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln()).collect();
        let sw: f64 = raw.iter().sum();
        let w: Vec<f64> = raw.iter().map(|x| x / sw).collect();
        let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let chin = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        // Active CMA: negative weights for the worse half.
        let raw_neg: Vec<f64> = (mu..lambda).map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln()).collect();
        let sum_neg: f64 = raw_neg.iter().map(|x| x.abs()).sum();
        let mueff_neg = if sum_neg > 0.0 { sum_neg.powi(2) / raw_neg.iter().map(|x| x * x).sum::<f64>() } else { 0.0 };
        let alpha_mu = 1.0 + c1 / cmu.max(1e-300);
        let alpha_mueff = 1.0 + 2.0 * mueff_neg / (mueff + 2.0);
        let alpha_posdef = (1.0 - c1 - cmu) / (nf * cmu.max(1e-300));
        let scale = alpha_mu.min(alpha_mueff).min(alpha_posdef) / sum_neg.max(1e-300);
        let mut w_all = w.clone();
        w_all.extend(raw_neg.iter().map(|x| x * scale));
        let mut c = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            c[i * n + i] = 1.0;
            b[i * n + i] = 1.0;
        }
        Cmaes {
            n,
            lambda,
            mu,
            w,
            w_all,
            mueff,
            cs,
            ds,
            cc,
            c1,
            cmu,
            chin,
            mean: mean.to_vec(),
            sigma,
            sigma0: sigma,
            c,
            b,
            d: vec![1.0; n],
            ps: vec![0.0; n],
            pc: vec![0.0; n],
            eigen_at: 0,
            iter: 0,
            pop: vec![0.0; lambda * n],
            z: vec![0.0; lambda * n],
            best_hist: vec![],
            median_hist: vec![],
            best_f: f64::INFINITY,
            best_x: mean.to_vec(),
        }
    }

    /// Replaces the covariance matrix (row-major, symmetric positive
    /// definite), e.g. by a quasi-Newton inverse Hessian for a warm start.
    /// It is rescaled to trace n, so `sigma` keeps its meaning.
    pub fn set_covariance(&mut self, c: &[f64]) {
        let n = self.n;
        assert_eq!(c.len(), n * n);
        let tr: f64 = (0..n).map(|i| c[i * n + i]).sum();
        if !(tr > 0.0 && tr.is_finite()) {
            return;
        }
        let mut ev = vec![0.0; n];
        let mut b = vec![0.0; n * n];
        let cs: Vec<f64> = c.iter().map(|t| t * n as f64 / tr).collect();
        sym_eig(&cs, n, &mut ev, &mut b);
        // Keep it well conditioned (condition number at most 1e14).
        let top = ev.iter().copied().fold(0.0, f64::max);
        if !(top > 0.0) {
            return;
        }
        for i in 0..n {
            self.d[i] = ev[i].max(top * 1e-14).sqrt();
        }
        for i in 0..n {
            for j in 0..n {
                self.c[i * n + j] = (0..n).map(|k| b[i * n + k] * self.d[k] * self.d[k] * b[j * n + k]).sum();
            }
        }
        self.b = b;
        self.eigen_at = self.iter;
    }

    /// Raises the eigenvalues of C to at least `1/cond` of the largest,
    /// keeping its eigenvectors and the step size.
    pub fn cap_condition(&mut self, cond: f64) {
        let n = self.n;
        let mut ev = vec![0.0; n];
        let mut b = vec![0.0; n * n];
        sym_eig(&self.c, n, &mut ev, &mut b);
        let top = ev.iter().copied().fold(0.0, f64::max);
        if !(top > 0.0) {
            return;
        }
        for i in 0..n {
            self.d[i] = ev[i].max(top / cond).sqrt();
        }
        for i in 0..n {
            for j in 0..n {
                self.c[i * n + j] = (0..n).map(|k| b[i * n + k] * self.d[k] * self.d[k] * b[j * n + k]).sum();
            }
        }
        self.b = b;
        self.eigen_at = self.iter;
    }

    /// Eigenvectors of C (columns, row-major) and the square roots of its
    /// eigenvalues.
    pub fn b_mat(&self) -> &[f64] {
        &self.b
    }

    pub fn d_vec(&self) -> &[f64] {
        &self.d
    }

    /// Replaces candidate `k` by `x` (after `ask`), with the step from the
    /// mean clipped to a typical length, as pycma's `inject` repairs it.
    pub fn inject(&mut self, k: usize, x: &[f64]) {
        let n = self.n;
        let mut z = vec![0.0; n];
        for j in 0..n {
            let s: f64 = (0..n).map(|i| self.b[i * n + j] * (x[i] - self.mean[i])).sum();
            z[j] = s / (self.sigma * self.d[j]);
        }
        let nz = z.iter().map(|t| t * t).sum::<f64>().sqrt();
        let nf = n as f64;
        let cap = nf.sqrt() + 2.0 * nf / (nf + 2.0);
        if nz > cap {
            z.iter_mut().for_each(|t| *t *= cap / nz);
        }
        for i in 0..n {
            let y: f64 = (0..n).map(|j| self.b[i * n + j] * self.d[j] * z[j]).sum();
            self.pop[k * n + i] = self.mean[i] + self.sigma * y;
        }
        self.z[k * n..(k + 1) * n].copy_from_slice(&z);
    }

    /// Samples `lambda` new candidates into `pop`.
    pub fn ask(&mut self, rng: &mut Rng) {
        let n = self.n;
        for k in 0..self.lambda {
            for i in 0..n {
                self.z[k * n + i] = rng.normal();
            }
            for i in 0..n {
                let mut y = 0.0;
                for j in 0..n {
                    y += self.b[i * n + j] * self.d[j] * self.z[k * n + j];
                }
                self.pop[k * n + i] = self.mean[i] + self.sigma * y;
            }
        }
    }

    /// Updates the distribution from the values of the candidates in `pop`.
    /// Returns why the run should stop, if it should.
    pub fn tell(&mut self, fit: &[f64]) -> Option<Stop> {
        self.update(fit, true)
    }

    /// As `tell`, for values that only rank the candidates (e.g. surrogate
    /// values): the best point is not taken from them; report true values
    /// with `observe`.
    pub fn tell_ranks(&mut self, fit: &[f64]) -> Option<Stop> {
        self.update(fit, false)
    }

    /// Records a truly evaluated point for `best_x` / `best_f`.
    pub fn observe(&mut self, x: &[f64], f: f64) {
        if f < self.best_f {
            self.best_f = f;
            self.best_x.copy_from_slice(x);
        }
    }

    fn update(&mut self, fit: &[f64], true_values: bool) -> Option<Stop> {
        let n = self.n;
        let nf = n as f64;
        self.iter += 1;
        let mut idx: Vec<usize> = (0..self.lambda).collect();
        idx.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
        if true_values && fit[idx[0]] < self.best_f {
            self.best_f = fit[idx[0]];
            self.best_x.copy_from_slice(&self.pop[idx[0] * n..(idx[0] + 1) * n]);
        }
        let old = self.mean.clone();
        for i in 0..n {
            self.mean[i] = (0..self.mu).map(|k| self.w[k] * self.pop[idx[k] * n + i]).sum();
        }
        // y_w = (m' - m)/sigma;  C^{-1/2} y_w = B z_w
        let yw: Vec<f64> = (0..n).map(|i| (self.mean[i] - old[i]) / self.sigma).collect();
        let zw: Vec<f64> = (0..n).map(|j| (0..self.mu).map(|k| self.w[k] * self.z[idx[k] * n + j]).sum()).collect();
        let csn = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for i in 0..n {
            let bz: f64 = (0..n).map(|j| self.b[i * n + j] * zw[j]).sum();
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + csn * bz;
        }
        let psn = self.ps.iter().map(|x| x * x).sum::<f64>().sqrt();
        let hsig = psn / (1.0 - (1.0 - self.cs).powi(2 * self.iter as i32)).sqrt() / self.chin < 1.4 + 2.0 / (nf + 1.0);
        let ccn = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        for i in 0..n {
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + if hsig { ccn * yw[i] } else { 0.0 };
        }
        let dh = if hsig { 0.0 } else { self.cc * (2.0 - self.cc) };
        // Rank-mu with active (negative) weights; negative-weight steps are
        // rescaled to Mahalanobis length sqrt(n).
        let wo: Vec<f64> = (0..self.lambda)
            .map(|r| {
                let w = self.w_all[r];
                if w >= 0.0 {
                    w
                } else {
                    let k = idx[r];
                    let z2: f64 = self.z[k * n..(k + 1) * n].iter().map(|t| t * t).sum();
                    w * nf / z2.max(1e-300)
                }
            })
            .collect();
        let wsum: f64 = self.w_all.iter().sum();
        let decay = 1.0 - self.c1 - self.cmu * wsum + self.c1 * dh;
        // Steps y_k = (x_k - m)/sigma, transposed (n rows of lambda), plain
        // and weighted, so that each rank-mu entry is one contiguous dot.
        let lam = self.lambda;
        let inv = 1.0 / self.sigma;
        let mut yt = vec![0.0; n * lam];
        let mut ywt = vec![0.0; n * lam];
        for (r, &k) in idx.iter().enumerate() {
            for i in 0..n {
                let y = (self.pop[k * n + i] - old[i]) * inv;
                yt[i * lam + r] = y;
                ywt[i * lam + r] = wo[r] * y;
            }
        }
        for i in 0..n {
            let wi = &ywt[i * lam..(i + 1) * lam];
            for j in 0..=i {
                let rank_mu: f64 = wi.iter().zip(&yt[j * lam..(j + 1) * lam]).map(|(a, b)| a * b).sum();
                let v = decay * self.c[i * n + j] + self.c1 * self.pc[i] * self.pc[j] + self.cmu * rank_mu;
                self.c[i * n + j] = v;
                self.c[j * n + i] = v;
            }
        }
        self.sigma *= ((self.cs / self.ds) * (psn / self.chin - 1.0)).min(1.0).exp();
        // Eigendecomposition every ~ 1/(c1+cmu)/n/10 iterations.
        let gap = ((self.lambda as f64) / (self.c1 + self.cmu) / nf / 10.0).max(1.0) as usize;
        if self.iter - self.eigen_at >= gap {
            self.eigen_at = self.iter;
            let mut ev = vec![0.0; n];
            sym_eig(&self.c, n, &mut ev, &mut self.b);
            for i in 0..n {
                self.d[i] = ev[i].max(1e-300).sqrt();
            }
        }
        // Termination (tutorial, Section B.3).
        let mut sorted = fit.to_vec();
        sorted.sort_by(f64::total_cmp);
        self.best_hist.push(sorted[0]);
        self.median_hist.push(sorted[sorted.len() / 2]);
        let hist = 10 + (30.0 * nf / self.lambda as f64).ceil() as usize;
        if self.best_hist.len() >= hist {
            let recent = &self.best_hist[self.best_hist.len() - hist..];
            let range = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max) - recent.iter().copied().fold(f64::INFINITY, f64::min);
            let span = sorted[sorted.len() - 1] - sorted[0];
            if range.max(span) < 1e-12 {
                return Some(Stop::TolFun);
            }
        }
        let sdmax = (0..n).map(|i| self.c[i * n + i].sqrt()).fold(0.0, f64::max);
        if self.sigma * sdmax.max(self.pc.iter().map(|x| x.abs()).fold(0.0, f64::max)) < 1e-11 * self.sigma0 {
            return Some(Stop::TolX);
        }
        let (dmin, dmax) = self.d.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
        if (dmax / dmin).powi(2) > 1e14 {
            return Some(Stop::Condition);
        }
        let ax = self.iter % n;
        if (0..n).all(|i| self.mean[i] == self.mean[i] + 0.1 * self.sigma * self.d[ax] * self.b[i * n + ax]) {
            return Some(Stop::NoEffect);
        }
        let it = self.best_hist.len();
        let stag = 120 + (30.0 * nf / self.lambda as f64) as usize;
        if it > stag {
            let w = (0.2 * it as f64) as usize;
            let med = |v: &[f64]| {
                let mut s = v.to_vec();
                let k = s.len() / 2;
                *s.select_nth_unstable_by(k, f64::total_cmp).1
            };
            if w >= 20
                && med(&self.best_hist[it - w..]) >= med(&self.best_hist[it - 2 * w.min(it / 2)..it - w])
                && med(&self.median_hist[it - w..]) >= med(&self.median_hist[it - 2 * w.min(it / 2)..it - w])
            {
                return Some(Stop::Stagnation);
            }
        }
        if self.iter as f64 > 100.0 + 150.0 * (nf + 3.0).powi(2) / (self.lambda as f64).sqrt() {
            return Some(Stop::MaxIter);
        }
        None
    }
}

/// Default population size `4 + floor(3 ln n)`.
pub fn default_lambda(n: usize) -> usize {
    4 + (3.0 * (n as f64).ln()).floor() as usize
}

/// IPOP-CMA-ES: restarts from a uniform point in `[lo, hi]` with sigma
/// `0.2 (hi - lo)` and doubling population, until `f` returns `true` from
/// `done` or `max_evals` evaluations are used. Returns the evaluations used.
pub fn ipop<F: FnMut(&[f64]) -> f64, D: FnMut() -> bool>(f: &mut F, mut done: D, lo: &[f64], hi: &[f64], max_evals: usize, seed: u64) -> usize {
    let n = lo.len();
    let mut rng = Rng::new(seed);
    let mut used = 0;
    let mut lambda = default_lambda(n);
    while used < max_evals && !done() {
        let x0: Vec<f64> = (0..n).map(|i| lo[i] + rng.uniform() * (hi[i] - lo[i])).collect();
        let sigma = 0.2 * (0..n).map(|i| hi[i] - lo[i]).fold(f64::INFINITY, f64::min);
        let mut es = Cmaes::new(&x0, sigma, lambda);
        let mut fit = vec![0.0; lambda];
        loop {
            if used + lambda > max_evals || done() {
                return used;
            }
            es.ask(&mut rng);
            for k in 0..lambda {
                fit[k] = f(&es.pop[k * n..(k + 1) * n]);
            }
            used += lambda;
            if es.tell(&fit).is_some() {
                break;
            }
        }
        lambda *= 2;
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigendecomposition_reconstructs() {
        for n in [1usize, 2, 7, 40] {
            let mut rng = Rng::new(n as u64);
            let g: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = (0..n).map(|k| g[i * n + k] * g[j * n + k]).sum();
                }
            }
            let (mut d, mut b) = (vec![0.0; n], vec![0.0; n * n]);
            for which in 0..2 {
                if which == 0 {
                    sym_eig(&a, n, &mut d, &mut b);
                } else {
                    jacobi_eig(&a, n, &mut d, &mut b);
                }
                let scale = a.iter().fold(0.0f64, |m, t| m.max(t.abs()));
                for i in 0..n {
                    for j in 0..n {
                        let r: f64 = (0..n).map(|k| b[i * n + k] * d[k] * b[j * n + k]).sum();
                        assert!((r - a[i * n + j]).abs() < 1e-10 * scale, "n={n} which={which}");
                    }
                }
            }
        }
    }

    #[test]
    fn cmaes_sphere_reference() {
        // Tutorial reference: about 1500 evaluations to 1e-8 on the 10-D sphere.
        let n = 10;
        let mut rng = Rng::new(1);
        let mut es = Cmaes::new(&vec![3.0; n], 2.0, default_lambda(n));
        let mut used = 0;
        let mut fit = vec![0.0; es.lambda];
        while es.best_f > 1e-8 && used < 5000 {
            es.ask(&mut rng);
            for k in 0..es.lambda {
                fit[k] = es.pop[k * n..(k + 1) * n].iter().map(|t| t * t).sum();
            }
            used += es.lambda;
            es.tell(&fit);
        }
        assert!(es.best_f <= 1e-8 && used < 2200, "used {used}");
    }
}
