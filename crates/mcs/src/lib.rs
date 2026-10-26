//! Multilevel Coordinate Search (MCS) for bound-constrained global
//! minimization, implemented from W. Huyer and A. Neumaier, "Global
//! optimization by multilevel coordinate search", J. Global Optim. 14
//! (1999) 331–355. Section references below are to that paper.
//!
//! The box tree, sweeps, splitting rules and shopping basket follow
//! Sections 3, 4 and 5.4. The local search of Section 5 (triple searches,
//! a bound-constrained QP and a line search) is replaced by BOBYQA
//! (Powell 2009), a trust-region method on the same kind of quadratic
//! models.
//!
//! The search is resumable: [`Mcs::run`] performs whole steps (an
//! initialization, one split, one local search) until a quota of
//! evaluations is used, and the next call continues where it stopped.
//!
//! Data layout: points and boxes live in flat arrays (`n` values per
//! entry), so a split copies contiguous memory and allocates nothing once
//! the arrays have grown. The record list of Section 3 is one min-heap per
//! level, keyed by the order-preserving integer image of `f`, with lazy
//! invalidation (an entry is live while its box is still at that level).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{BuildHasherDefault, Hasher};

mod split;

/// Golden section ratio q = (sqrt(5) - 1) / 2 (Section 3).
const Q: f64 = 0.618_033_988_749_894_9;

/// Final radius of the coarse local stage, relative to the initial one.
const COARSE: f64 = 1e-3;
/// sigma of the single-linkage critical distance (Rinnooy Kan & Timmer use 4).
const MLSL_SIGMA: f64 = 4.0;
/// Basket points a candidate is screened against (nearest first).
const SCREEN: usize = 3;

/// Settings; [`Config::new`] gives the paper's defaults (Section 7.3).
#[derive(Clone, Debug)]
pub struct Config {
    /// Number of levels; boxes at level `smax` are not split any more.
    pub smax: usize,
    /// Run local searches from shopping-basket candidates.
    pub local: bool,
    /// Evaluation budget of one local search.
    pub local_budget: usize,
    /// Stop as soon as a value `<= f_target` is found.
    pub f_target: f64,
    /// Raise `smax` when every box has reached it, instead of stopping.
    pub grow: bool,
    /// Initial local-search radius: this fraction of the candidate box's
    /// longest side, but at least `rho_min_frac` of the narrowest bound width.
    pub rho_frac: f64,
    pub rho_min_frac: f64,
}

impl Config {
    pub fn new(n: usize) -> Self {
        Config { smax: 5 * n + 10, local: true, local_budget: 50 * n, f_target: f64::NEG_INFINITY, grow: true, rho_frac: 0.5, rho_min_frac: 1e-6 }
    }
}

/// Order-preserving map from `f64` to `u64` (NaN sorts last, -0 = +0).
#[inline(always)]
fn key(f: f64) -> u64 {
    let f = if f.is_nan() { f64::INFINITY } else { f + 0.0 };
    let b = f.to_bits();
    // Flip all bits of negatives, only the sign bit of non-negatives.
    b ^ (((b as i64) >> 63) as u64 | (1 << 63))
}

/// One splitting event: the coordinate and the points along it (value of
/// the coordinate and f). At most three points (the initialization list
/// has three values; a regular split two), stored inline so a history walk
/// touches one cache line per step.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Event {
    pts: [(f64, f64); 3],
    /// Event that created the box that was split (`NONE` for the root).
    parent: u32,
    coord: u16,
    len: u16,
}

const NONE: u32 = u32::MAX;

/// Initialization list of one coordinate (Section 3): values `x^l_i`, their
/// function values `f^l_i` and the index `l_i` of the value in `x0`.
#[derive(Clone, Default)]
struct InitList {
    x: Vec<f64>,
    f: Vec<f64>,
    l0: usize,
}

/// Why [`Mcs::run`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The quota was used up; call `run` again to continue.
    Quota,
    /// A value `<= f_target` was found.
    Target,
    /// Every box has reached level `smax`: the search is complete.
    Finished,
}

/// The state of one MCS search.
pub struct Mcs {
    n: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    cfg: Config,
    // Evaluated points (n per point) and their values.
    xs: Vec<f64>,
    fs: Vec<f64>,
    // Boxes, structure of arrays: level (0 = split), base point, opposite
    // point (n per box), split count per coordinate (n per box), creating event.
    level: Vec<u16>,
    base: Vec<u32>,
    opp: Vec<f64>,
    nsplit: Vec<u16>,
    born: Vec<u32>,
    events: Vec<Event>,
    heaps: Vec<BinaryHeap<Reverse<u128>>>,
    /// Live (filed, unsplit) boxes per level, and a bit per non-empty level.
    count: Vec<u32>,
    occupied: [u64; 4],
    filed: Vec<bool>,
    /// Cached expected gain per box: (coordinate, gain, split value); the
    /// model depends only on the box's fixed history.
    gain: Vec<(u32, f64, f64)>,
    init: Vec<InitList>,
    /// Coordinate ranks by estimated variability (0 = most variable).
    pi: Vec<usize>,
    initialized: bool,
    finished: bool,
    /// Current level of the sweep in progress (0 = no sweep in progress).
    sweep_s: usize,
    /// Shopping-basket candidates: (point, box) of boxes that reached smax.
    candidates: Vec<(u32, u32)>,
    basket: Vec<u32>,
    searched: Vec<bool>,
    best: usize,
    used: usize,
    /// Evaluations per phase: initialization, splits, basket screening, local search.
    phase: usize,
    pub stats: [usize; 4],
    /// Coarsely converged best local minimum awaiting refinement: point,
    /// radius reached, and the sweep count when it was found.
    pending: Option<(usize, f64, usize)>,
    sweeps: usize,
    target_hit: bool,
    /// Hash of every evaluated point's bits -> its index (exact duplicates
    /// are never evaluated twice).
    seen: std::collections::HashMap<u64, u32, BuildHasherDefault<IdHasher>>,
    // Scratch.
    xbuf: Vec<f64>,
    ybuf: Vec<f64>,
    nsbuf: Vec<u16>,
    cnt_buf: Vec<u8>,
    pts_buf: Vec<(f64, f64)>,
}

impl Mcs {
    /// A search over the box `[u, v]` (finite bounds, `u < v`).
    pub fn new(u: &[f64], v: &[f64], cfg: Config) -> Self {
        let n = u.len();
        assert!(n > 0 && v.len() == n, "bounds must have equal, nonzero length");
        assert!((0..n).all(|i| u[i].is_finite() && v[i].is_finite() && u[i] < v[i]), "bounds must be finite with u < v");
        assert!(cfg.smax >= 3 && cfg.smax < 256, "smax must be in 3..256");
        assert!(n < u16::MAX as usize, "dimension too large");
        Mcs {
            n,
            u: u.to_vec(),
            v: v.to_vec(),
            heaps: vec![BinaryHeap::new(); cfg.smax + 1],
            count: vec![0; cfg.smax + 1],
            occupied: [0; 4],
            filed: Vec::new(),
            gain: Vec::new(),
            cfg,
            xs: Vec::new(),
            fs: Vec::new(),
            level: Vec::new(),
            base: Vec::new(),
            opp: Vec::new(),
            nsplit: Vec::new(),
            born: Vec::new(),
            events: Vec::new(),
            init: Vec::new(),
            pi: Vec::new(),
            initialized: false,
            finished: false,
            sweep_s: 0,
            candidates: Vec::new(),
            basket: Vec::new(),
            searched: Vec::new(),
            best: 0,
            used: 0,
            phase: 0,
            stats: [0; 4],
            pending: None,
            sweeps: 0,
            target_hit: false,
            seen: Default::default(),
            xbuf: vec![0.0; n],
            ybuf: vec![0.0; n],
            nsbuf: vec![0; n],
            cnt_buf: Vec::new(),
            pts_buf: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// Evaluations used so far.
    pub fn evaluations(&self) -> usize {
        self.used
    }

    /// Best point and value so far (`None` before the first evaluation).
    pub fn best(&self) -> Option<(&[f64], f64)> {
        (!self.fs.is_empty()).then(|| (self.point(self.best), self.fs[self.best]))
    }

    /// All evaluated points and values, in evaluation order.
    pub fn archive(&self) -> (&[f64], &[f64]) {
        (&self.xs, &self.fs)
    }

    /// Number of local minimizers in the shopping basket.
    pub fn basket_len(&self) -> usize {
        self.basket.len()
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    #[inline(always)]
    fn point(&self, k: usize) -> &[f64] {
        &self.xs[k * self.n..(k + 1) * self.n]
    }

    /// Evaluates `f` at `x`, records the point and returns its index. A
    /// point evaluated before (bit-identical coordinates) is not evaluated
    /// again: its index is returned.
    fn eval<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, x: &[f64]) -> usize {
        let h = point_hash(x);
        if let Some(&k) = self.seen.get(&h) {
            let k = k as usize;
            if self.point(k).iter().zip(x).all(|(a, b)| a.to_bits() == b.to_bits()) {
                return k;
            }
        }
        let fx = f(x);
        self.seen.insert(h, self.fs.len() as u32);
        self.xs.extend_from_slice(x);
        self.fs.push(fx);
        self.searched.push(false);
        let k = self.fs.len() - 1;
        self.used += 1;
        self.stats[self.phase] += 1;
        if k == 0 || fx < self.fs[self.best] {
            self.best = k;
        }
        if fx <= self.cfg.f_target {
            self.target_hit = true;
        }
        k
    }

    fn fbest(&self) -> f64 {
        self.fs[self.best]
    }

    /// Runs until about `quota` more evaluations are used (whole steps
    /// only: a split or local search in progress may overshoot by its own
    /// size, never beyond `hard_cap` evaluations in total), the target is
    /// hit, or the search is complete.
    pub fn run<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, quota: usize, hard_cap: usize) -> Status {
        let stop = self.used.saturating_add(quota).min(hard_cap);
        // Room for the points and boxes this call will create (a split
        // creates at most three boxes per evaluation).
        let more = stop - self.used.min(stop);
        self.xs.reserve(more * self.n);
        self.fs.reserve(more);
        self.searched.reserve(more);
        self.seen.reserve(more);
        let boxes = 3 * more + 8 * self.n;
        self.level.reserve(boxes);
        self.base.reserve(boxes);
        self.born.reserve(boxes);
        self.filed.reserve(boxes);
        self.gain.reserve(boxes);
        self.opp.reserve(boxes * self.n);
        self.nsplit.reserve(boxes * self.n);
        self.events.reserve(more + self.n);
        loop {
            if self.target_hit {
                return Status::Target;
            }
            if self.finished {
                return Status::Finished;
            }
            if self.used >= stop {
                return Status::Quota;
            }
            let room = hard_cap - self.used;
            if !self.initialized {
                // 1 + 2n evaluations for the three-point list.
                if room < 1 + 2 * self.n {
                    return Status::Quota;
                }
                self.initialize(f);
            } else if !self.candidates.is_empty() && self.sweep_s == 0 {
                if !self.basket_step(f, hard_cap) {
                    return Status::Quota;
                }
            } else if self.sweep_s == 0 && self.pending.is_some_and(|p| p.2 < self.sweeps) {
                if !self.refine(f, hard_cap) {
                    return Status::Quota;
                }
            } else {
                self.phase = 1;
                if !self.sweep_step(f, hard_cap) {
                    // The next split needs more evaluations than the cap leaves.
                    return Status::Quota;
                }
            }
        }
    }

    // ----- Section 3: initialization -----

    fn initialize<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F) {
        let n = self.n;
        // Simple initialization list (Section 7.3): u, midpoint, v; x0 = midpoint.
        let mut x: Vec<f64> = (0..n).map(|i| 0.5 * (self.u[i] + self.v[i])).collect();
        let k0 = self.eval(f, &x);
        let mut fstar = self.fs[k0];
        let mut kstar = k0;
        self.init = vec![InitList::default(); n];
        // Points of each coordinate's line, to build the boxes below.
        let mut lines: Vec<Vec<(f64, f64, u32)>> = Vec::with_capacity(n);
        for i in 0..n {
            let vals = [self.u[i], x[i], self.v[i]];
            let mut line = Vec::with_capacity(3);
            for (l, &xi) in vals.iter().enumerate() {
                if l == 1 {
                    line.push((xi, fstar, kstar as u32));
                    continue;
                }
                let keep = x[i];
                x[i] = xi;
                let k = self.eval(f, &x);
                x[i] = keep;
                line.push((xi, self.fs[k], k as u32));
            }
            let lstar = (0..3).min_by(|&a, &b| line[a].1.total_cmp(&line[b].1)).unwrap();
            self.init[i] = InitList { x: vals.to_vec(), f: line.iter().map(|p| p.1).collect(), l0: 1 };
            x[i] = line[lstar].0;
            fstar = line[lstar].1;
            kstar = line[lstar].2 as usize;
            lines.push(line);
        }
        // Coordinate priorities from the variability of the quadratic
        // interpolants of each line (Section 3).
        let var: Vec<f64> = lines.iter().map(|l| split::variability(l)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| var[b].total_cmp(&var[a]));
        self.pi = vec![0; n];
        for (r, &i) in order.iter().enumerate() {
            self.pi[i] = r;
        }
        // Root box: base x0, opposite the farthest corner, level 1.
        let x0: Vec<f64> = (0..n).map(|i| 0.5 * (self.u[i] + self.v[i])).collect();
        let y0: Vec<f64> = (0..n).map(|i| if x0[i] - self.u[i] >= self.v[i] - x0[i] { self.u[i] } else { self.v[i] }).collect();
        let mut cur = self.new_box(k0 as u32, &y0, &vec![0; n], 1, NONE);
        // Split the current box along each coordinate in turn.
        for (i, line) in lines.iter().enumerate() {
            let s = self.level[cur] as usize;
            let children = self.split_line(cur, i, line, s);
            if i + 1 == n {
                break;
            }
            // Next current box: the child whose base is x* (after this
            // coordinate) on the side of the line's quadratic minimizer.
            let lstar = (0..line.len()).min_by(|&a, &b| line[a].1.total_cmp(&line[b].1)).unwrap();
            let xs_i = line[lstar].0;
            let side = split::parabola_side(line, lstar);
            let mut pick = None;
            for &c in &children {
                if self.base[c] == line[lstar].2 {
                    let (lo, hi) = self.interval(c, i);
                    let on_right = lo >= xs_i;
                    let ok = if hi <= xs_i {
                        side <= 0.0
                    } else if on_right {
                        side >= 0.0
                    } else {
                        true
                    };
                    if ok || pick.is_none() {
                        pick = Some(c);
                        if ok {
                            break;
                        }
                    }
                }
            }
            cur = pick.expect("x* lies in a child box");
        }
        // Every non-split box enters its level's record heap.
        for b in 0..self.level.len() {
            self.file(b);
        }
        self.initialized = true;
    }

    // ----- boxes -----

    fn new_box(&mut self, base: u32, opp: &[f64], nsplit: &[u16], level: usize, born: u32) -> usize {
        let b = self.level.len();
        self.level.push(level as u16);
        self.base.push(base);
        self.opp.extend_from_slice(opp);
        self.nsplit.extend_from_slice(nsplit);
        self.born.push(born);
        self.filed.push(false);
        self.gain.push((u32::MAX - 1, 0.0, 0.0));
        b
    }

    /// Takes box `b` out of its level (it was split or is moving up).
    #[inline]
    pub(crate) fn unfile(&mut self, b: usize) {
        if self.filed[b] {
            self.filed[b] = false;
            let s = self.level[b] as usize;
            self.count[s] -= 1;
            if self.count[s] == 0 {
                self.occupied[s >> 6] &= !(1u64 << (s & 63));
            }
        }
    }

    /// Lowest non-empty level `>= from` below smax, by bit scan.
    #[inline]
    fn next_level(&self, from: usize) -> Option<usize> {
        let mut w = from >> 6;
        if w >= 4 {
            return None;
        }
        let mut bits = self.occupied[w] & (u64::MAX << (from & 63));
        loop {
            if bits != 0 {
                let s = (w << 6) | bits.trailing_zeros() as usize;
                return (s < self.cfg.smax).then_some(s);
            }
            w += 1;
            if w == 4 {
                return None;
            }
            bits = self.occupied[w];
        }
    }

    /// The `i`th coordinate interval of box `b`.
    #[inline(always)]
    fn interval(&self, b: usize, i: usize) -> (f64, f64) {
        let xi = self.xs[self.base[b] as usize * self.n + i];
        let yi = self.opp[b * self.n + i];
        if xi <= yi { (xi, yi) } else { (yi, xi) }
    }

    /// Puts a non-split box into its level's record heap, or into the
    /// candidate list at level smax.
    #[cfg_attr(feature = "profile", inline(never))]
    fn file(&mut self, b: usize) {
        let s = self.level[b] as usize;
        if s == 0 {
            return;
        }
        if s >= self.cfg.smax {
            self.candidates.push((self.base[b], b as u32));
            return;
        }
        // Ties in f go to the lower box number, the older box.
        let k = ((key(self.fs[self.base[b] as usize]) as u128) << 64) | b as u128;
        self.heaps[s].push(Reverse(k));
        self.filed[b] = true;
        self.count[s] += 1;
        self.occupied[s >> 6] |= 1u64 << (s & 63);
    }

    /// All boxes have reached `smax`: raise it (the convergence theorem of
    /// Section 6 holds as smax grows without bound) and put the boxes at the
    /// old cap back into play. False if smax cannot grow further.
    fn grow_smax(&mut self) -> bool {
        let old = self.cfg.smax;
        let new = (old + 5 * self.n + 10).min(255);
        if new <= old || !self.cfg.grow {
            return false;
        }
        self.cfg.smax = new;
        self.heaps.resize(new + 1, BinaryHeap::new());
        self.count.resize(new + 1, 0);
        for b in 0..self.level.len() {
            if self.level[b] as usize == old {
                self.file(b);
            }
        }
        true
    }

    /// Removes and returns the best box of level `s`. The sweep always
    /// works on the top box, so heaps never hold stale entries.
    #[inline]
    fn take_top(&mut self, s: usize) -> Option<usize> {
        let Reverse(k) = self.heaps[s].pop()?;
        let b = (k & u64::MAX as u128) as usize;
        debug_assert!(self.filed[b] && self.level[b] as usize == s);
        self.filed[b] = false;
        self.count[s] -= 1;
        if self.count[s] == 0 {
            self.occupied[s >> 6] &= !(1u64 << (s & 63));
        }
        Some(b)
    }

    fn new_event(&mut self, parent_box: usize, coord: usize, pts: &[(f64, f64)]) -> u32 {
        let e = self.events.len() as u32;
        let mut ev = Event { pts: [(0.0, 0.0); 3], parent: self.born[parent_box], coord: coord as u16, len: pts.len() as u16 };
        ev.pts[..pts.len()].copy_from_slice(pts);
        self.events.push(ev);
        e
    }

    // ----- Section 3: sweeps -----

    /// One step of a sweep; false if the step needed more room than left.
    #[cfg_attr(feature = "profile", inline(never))]
    fn sweep_step<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, hard_cap: usize) -> bool {
        if self.sweep_s == 0 {
            // Step 1: start at the lowest non-empty level.
            match self.next_level(1) {
                Some(s) => {
                    self.sweep_s = s;
                    self.sweeps += 1;
                }
                None => {
                    if self.candidates.is_empty() && !self.grow_smax() {
                        self.finished = true;
                    }
                    return true;
                }
            }
        }
        let s = self.sweep_s;
        if let Some(b) = self.take_top(s) {
            // Step 2: split the best box of this level, or raise its level.
            match self.try_split(f, b, hard_cap) {
                split::Split::Done => {}
                split::Split::NoRoom => {
                    // Put it back; the step is retried in the next call.
                    self.file(b);
                    return false;
                }
                split::Split::Declined => {
                    self.unfile(b);
                    self.level[b] += 1;
                    self.file(b);
                }
            }
        }
        // Step 3: next non-empty level, or end of sweep.
        self.sweep_s = self.next_level(s + 1).unwrap_or(0);
        true
    }

    // ----- Section 5.4: shopping basket -----

    /// One shopping-basket step; false (leaving the candidate in place) if
    /// a local search would not fit under `hard_cap`.
    fn basket_step<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, hard_cap: usize) -> bool {
        if self.cfg.local && hard_cap.saturating_sub(self.used) < 2 * self.n + 3 + 2 * SCREEN {
            return false;
        }
        // Candidates in ascending order of f; process one per step.
        let fs = &self.fs;
        self.candidates.sort_by(|a, b| fs[b.0 as usize].total_cmp(&fs[a.0 as usize]));
        let Some((c, cbox)) = self.candidates.pop() else { return true };
        let (c, cbox) = (c as usize, cbox as usize);
        if !self.cfg.local || self.searched[c] {
            return true;
        }
        self.searched[c] = true;
        if self.linked(c) {
            return true;
        }
        self.phase = 2;
        let Some(start) = self.screen(f, c, hard_cap) else { return true };
        if self.target_hit {
            return true;
        }
        let room = hard_cap.saturating_sub(self.used);
        let budget = self.cfg.local_budget.min(room);
        self.phase = 3;
        let Some(xloc) = self.local_search(f, start, cbox, budget) else { return true };
        self.phase = 2;
        if self.target_hit {
            return true;
        }
        if self.screen(f, xloc, hard_cap).is_some() || self.basket.is_empty() {
            self.searched[xloc] = true;
            self.basket.push(xloc as u32);
        }
        true
    }

    /// Multi-level single linkage (Rinnooy Kan & Timmer, 1987): true if a
    /// better evaluated point lies within the critical distance
    /// r_k = pi^(-1/2) (Gamma(1 + n/2) vol sigma ln k / k)^(1/n) of point
    /// `c`, which then needs no local search of its own.
    fn linked(&self, c: usize) -> bool {
        let n = self.n;
        let k = self.fs.len() as f64;
        let vol_ln: f64 = (0..n).map(|i| (self.v[i] - self.u[i]).ln()).sum();
        let r_ln = (ln_gamma(1.0 + n as f64 / 2.0) + vol_ln + (MLSL_SIGMA * k.ln() / k).ln()) / n as f64 - 0.5 * std::f64::consts::PI.ln();
        let r2 = (2.0 * r_ln).exp();
        let fc = self.fs[c];
        let xc = self.point(c);
        (0..self.fs.len()).any(|j| self.fs[j] < fc && dist2(xc, self.point(j)) <= r2)
    }

    /// Steps 1–3 of Section 5.4: compares point `k` with the basket points
    /// that are at least as good, nearest first, by two evaluations on the
    /// segment towards each. Returns the (possibly moved) start point, or
    /// `None` if `k` lies in the basin of a basket point.
    fn screen<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, k: usize, hard_cap: usize) -> Option<usize> {
        let n = self.n;
        let mut x = k;
        let mut ws: Vec<(f64, u32)> =
            self.basket.iter().filter(|&&w| self.fs[w as usize] <= self.fs[k]).map(|&w| (dist2(self.point(k), self.point(w as usize)), w)).collect();
        ws.sort_by(|a, b| a.0.total_cmp(&b.0));
        ws.truncate(SCREEN);
        for &(_, w) in &ws {
            if hard_cap.saturating_sub(self.used) < 2 || self.target_hit {
                return Some(x);
            }
            let w = w as usize;
            let fx = self.fs[x];
            for j in 0..n {
                self.xbuf[j] = self.xs[x * n + j] + (self.xs[w * n + j] - self.xs[x * n + j]) / 3.0;
            }
            let xb = std::mem::take(&mut self.xbuf);
            let k1 = self.eval(f, &xb);
            self.xbuf = xb;
            if self.fs[k1] > fx {
                continue;
            }
            for j in 0..n {
                self.xbuf[j] = self.xs[x * n + j] + 2.0 * (self.xs[w * n + j] - self.xs[x * n + j]) / 3.0;
            }
            let xb = std::mem::take(&mut self.xbuf);
            let k2 = self.eval(f, &xb);
            self.xbuf = xb;
            let (f1, f2, fw) = (self.fs[k1], self.fs[k2], self.fs[w]);
            if f2 > f1.max(fw) {
                if f1 < fx {
                    x = k1;
                }
            } else if f1.min(f2) < fw {
                x = if f1 <= f2 { k1 } else { k2 };
            } else {
                return None;
            }
        }
        Some(x)
    }

    /// BOBYQA from point `k` inside `[u, v]`; returns the index of the best
    /// point it evaluated.
    fn local_search<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, k: usize, b: usize, budget: usize) -> Option<usize> {
        let n = self.n;
        if budget < 2 * n + 3 {
            return None;
        }
        let width = (0..n).map(|i| self.v[i] - self.u[i]).fold(f64::INFINITY, f64::min);
        // Initial trust radius: the scale of the candidate's own box.
        let side = (0..n)
            .map(|i| {
                let (lo, hi) = self.interval(b, i);
                hi - lo
            })
            .fold(0.0, f64::max);
        let rho0 = (self.cfg.rho_frac * side).clamp(self.cfg.rho_min_frac.min(0.1) * width, 0.1 * width);
        let floor = 1e-10 * width;
        let f_before = self.fbest();
        let start = self.fs.len();
        // Stage 1: coarse convergence, cheap in basins that are not global.
        let rho1 = (COARSE * rho0).max(floor);
        let mut x = self.point(k).to_vec();
        self.bobyqa(f, &mut x, rho0, rho1, budget)?;
        let p = (start..self.fs.len()).min_by(|&a, &b| self.fs[a].total_cmp(&self.fs[b]))?;
        // Stage 2 (refinement to full accuracy) is deferred: it runs only if
        // this basin is still the best after another full sweep.
        if !self.target_hit && rho1 > floor && self.fs[p] <= f_before {
            self.pending = Some((p, rho1, self.sweeps));
        }
        (start..self.fs.len()).min_by(|&a, &b| self.fs[a].total_cmp(&self.fs[b]))
    }

    /// Refines the pending local minimum if it is still the best point;
    /// false (keeping it pending) if the refinement would not fit.
    fn refine<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, hard_cap: usize) -> bool {
        if hard_cap.saturating_sub(self.used) < 2 * self.n + 3 {
            return false;
        }
        let Some((p, rho1, _)) = self.pending.take() else { return true };
        if self.fs[p] > self.fbest() {
            return true;
        }
        let n = self.n;
        let width = (0..n).map(|i| self.v[i] - self.u[i]).fold(f64::INFINITY, f64::min);
        let budget = self.cfg.local_budget.min(hard_cap.saturating_sub(self.used));
        self.phase = 3;
        let mut x = self.point(p).to_vec();
        self.bobyqa(f, &mut x, (10.0 * rho1).min(0.1 * width), 1e-10 * width, budget);
        self.phase = 1;
        true
    }

    /// One BOBYQA solve from `x` with the given radii and budget.
    fn bobyqa<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, x: &mut [f64], rho_begin: f64, rho_end: f64, budget: usize) -> Option<()> {
        let n = self.n;
        if budget < 2 * n + 3 {
            return None;
        }
        let mut cfg = bobyqa::Config::new(n);
        cfg.rho_begin = rho_begin;
        cfg.rho_end = rho_end.min(rho_begin);
        cfg.max_fun = budget;
        cfg.f_target = self.cfg.f_target;
        let mut solver = bobyqa::Bobyqa::new(n, cfg).ok()?;
        let (u, v) = (self.u.clone(), self.v.clone());
        solver.minimize(|y| self.eval_ret(f, y), x, &u, &v);
        Some(())
    }

    fn eval_ret<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, y: &[f64]) -> f64 {
        let k = self.eval(f, y);
        self.fs[k]
    }
}

/// 64-bit hash of a point's coordinate bits (multiply-rotate mixing; the
/// map verifies candidates bit for bit, so collisions only cost a miss).
#[inline]
fn point_hash(x: &[f64]) -> u64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for &t in x {
        h = (h ^ t.to_bits()).wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(29);
    }
    h ^ (h >> 32)
}

/// Pass-through hasher for keys that already are hashes.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, 9 terms; ~15 digits).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[inline]
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum()
}
