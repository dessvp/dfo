//! Splitting (Section 4) and the line splits shared with the
//! initialization (Section 3).

use crate::{InitList, Mcs, Q};

/// Quadratic through three points, as `f(t) = a + b (t - t0) + c (t - t0)^2`.
#[inline]
fn parabola(p: [(f64, f64); 3]) -> (f64, f64, f64) {
    let (t0, f0) = p[0];
    let d1 = (p[1].1 - f0) / (p[1].0 - t0);
    let d2 = (p[2].1 - f0) / (p[2].0 - t0);
    let c = (d2 - d1) / (p[2].0 - p[1].0);
    let b = d1 - c * (p[1].0 - t0);
    (f0, b, c)
}

/// Range of the quadratic interpolants through consecutive triples of a
/// line, as a measure of the variability of f along it (Section 3).
pub(crate) fn variability(line: &[(f64, f64, u32)]) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for w in line.windows(3) {
        let pts = [(w[0].0, w[0].1), (w[1].0, w[1].1), (w[2].0, w[2].1)];
        let (a, b, c) = parabola(pts);
        let t0 = pts[0].0;
        let mut ext = |t: f64| {
            let d = t - t0;
            let v = a + d * (b + c * d);
            lo = lo.min(v);
            hi = hi.max(v);
        };
        ext(pts[0].0);
        ext(pts[2].0);
        if c != 0.0 {
            let tv = t0 - b / (2.0 * c);
            if tv > pts[0].0 && tv < pts[2].0 {
                ext(tv);
            }
        }
    }
    if hi >= lo { hi - lo } else { 0.0 }
}

/// Which side of the best point of a line the minimizer of the local
/// quadratic lies on: negative left, positive right.
pub(crate) fn parabola_side(line: &[(f64, f64, u32)], l: usize) -> f64 {
    let m = line.len();
    let c = l.clamp(1, m - 2);
    let pts = [(line[c - 1].0, line[c - 1].1), (line[c].0, line[c].1), (line[c + 1].0, line[c + 1].1)];
    let (_, b, cc) = parabola(pts);
    let t = line[l].0;
    if cc > 0.0 {
        let tv = pts[0].0 - b / (2.0 * cc);
        return tv - t;
    }
    // Not convex: towards the better neighbour.
    let left = if l > 0 { line[l - 1].1 } else { f64::INFINITY };
    let right = if l + 1 < m { line[l + 1].1 } else { f64::INFINITY };
    if left < right { -1.0 } else { 1.0 }
}

/// Safeguarded far end of a split interval: the three cases of eq. (4).
#[inline]
fn subint(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let near_zero = 1000.0 * ax < 1.0;
    let far_small = near_zero && ay > 1000.0;
    let far_large = !near_zero && ay > 1000.0 * ax;
    match (far_small, far_large) {
        (true, _) => y.signum(),
        (_, true) => 10.0 * y.signum() * ax,
        _ => y,
    }
}

/// Outcome of offering a box for splitting.
#[derive(PartialEq, Eq)]
pub(crate) enum Split {
    Done,
    Declined,
    NoRoom,
}

impl Mcs {
    /// Splits box `cur` (level `s`) along coordinate `i` at the points of a
    /// line through its base (`line`: coordinate value, f, point index,
    /// ascending in the coordinate), with golden-section points between
    /// neighbours (Section 3). Returns the children (not yet filed).
    pub(crate) fn split_line(&mut self, cur: usize, i: usize, line: &[(f64, f64, u32)], s: usize) -> Vec<usize> {
        let n = self.n;
        let smax = self.cfg.smax;
        let (lo, hi) = self.interval(cur, i);
        let pts: Vec<(f64, f64)> = line.iter().map(|p| (p.0, p.1)).collect();
        let ev = self.new_event(cur, i, &pts);
        self.ybuf.copy_from_slice(&self.opp[cur * n..(cur + 1) * n]);
        let mut ns: Vec<u16> = self.nsplit[cur * n..(cur + 1) * n].to_vec();
        ns[i] += 1;
        let (s1, s2) = ((s + 1).min(smax), (s + 2).min(smax));
        let mut kids = Vec::with_capacity(2 * line.len());
        let push = |me: &mut Mcs, base: u32, other_end: f64, lev: usize, kids: &mut Vec<usize>| {
            let mut y = me.ybuf.clone();
            y[i] = other_end;
            kids.push(me.new_box(base, &y, &ns, lev, ev));
        };
        let first = line[0];
        if first.0 > lo {
            push(self, first.2, lo, s1, &mut kids);
        }
        for w in line.windows(2) {
            let (a, b) = (w[0], w[1]);
            // The part next to the smaller value gets the larger fraction
            // (on a tie the left part: the paper leaves ties open).
            let left_big = a.1 <= b.1;
            let z = if left_big { a.0 + Q * (b.0 - a.0) } else { a.0 + Q * Q * (b.0 - a.0) };
            push(self, a.2, z, if left_big { s1 } else { s2 }, &mut kids);
            push(self, b.2, z, if left_big { s2 } else { s1 }, &mut kids);
        }
        let last = line[line.len() - 1];
        if last.0 < hi {
            push(self, last.2, hi, s1, &mut kids);
        }
        self.unfile(cur);
        self.level[cur] = 0;
        kids
    }

    /// Section 4: split box `b` by rank or by expected gain, or decline.
    #[cfg_attr(feature = "profile", inline(never))]
    pub(crate) fn try_split<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, b: usize, hard_cap: usize) -> Split {
        let n = self.n;
        let s = self.level[b] as usize;
        let room = hard_cap.saturating_sub(self.used);
        let xb = self.base[b] as usize;
        let ns = &self.nsplit[b * n..(b + 1) * n];
        let m = *ns.iter().min().unwrap() as usize;
        if s > 2 * n * (m + 1) {
            // 4.1: least-split coordinate, most variable first.
            let i = (0..n).filter(|&j| ns[j] as usize == m).min_by_key(|&j| self.pi[j]).unwrap();
            if m == 0 {
                return self.split_init(f, b, i, room);
            }
            let (xi, yi) = (self.xs[xb * n + i], self.opp[b * n + i]);
            let z = xi + 2.0 / 3.0 * (subint(xi, yi) - xi);
            return self.split_at(f, b, i, z, false, room);
        }
        // 4.2: separable quadratic model from the box history (cached).
        let (i, gain, z) = if self.gain[b].0 == u32::MAX - 1 {
            let g = self.expected_gain(b);
            self.gain[b] = (g.0 as u32, g.1, g.2);
            g
        } else {
            let g = self.gain[b];
            (if g.0 == u32::MAX { usize::MAX } else { g.0 as usize }, g.1, g.2)
        };
        if i == usize::MAX || self.fs[xb] + gain >= self.fbest() {
            return Split::Declined;
        }
        if self.nsplit[b * n + i] == 0 { self.split_init(f, b, i, room) } else { self.split_at(f, b, i, z, true, room) }
    }

    /// Split along a never-split coordinate at the initialization list.
    fn split_init<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, b: usize, i: usize, room: usize) -> Split {
        let n = self.n;
        let list: InitList = self.init[i].clone();
        if room < list.x.len() - 1 {
            return Split::NoRoom;
        }
        let xb = self.base[b] as usize;
        let mut x = self.point(xb).to_vec();
        let mut line = Vec::with_capacity(list.x.len());
        for (l, &t) in list.x.iter().enumerate() {
            if l == list.l0 {
                line.push((x[i], self.fs[xb], xb as u32));
            } else {
                let keep = x[i];
                x[i] = t;
                let k = self.eval(f, &x);
                x[i] = keep;
                line.push((t, self.fs[k], k as u32));
            }
        }
        line.sort_by(|a, b| a.0.total_cmp(&b.0));
        let s = self.level[b] as usize;
        let kids = self.split_line(b, i, &line, s);
        for c in kids {
            self.file(c);
        }
        let _ = n;
        Split::Done
    }

    /// Split along coordinate `i` at `z` (between the base and the opposite
    /// point) and at the golden-section point between base and `z`.
    #[cfg_attr(feature = "profile", inline(never))]
    fn split_at<F: FnMut(&[f64]) -> f64>(&mut self, f: &mut F, b: usize, i: usize, z: f64, by_gain: bool, room: usize) -> Split {
        let n = self.n;
        if room < 1 {
            return Split::NoRoom;
        }
        let smax = self.cfg.smax;
        let s = self.level[b] as usize;
        let xb = self.base[b] as usize;
        let (xi, yi) = (self.xs[xb * n + i], self.opp[b * n + i]);
        if z == xi || !z.is_finite() {
            return Split::Declined;
        }
        let mut x = std::mem::take(&mut self.xbuf);
        x.copy_from_slice(&self.xs[xb * n..(xb + 1) * n]);
        x[i] = z;
        let k = self.eval(f, &x);
        self.xbuf = x;
        let (fx, fz) = (self.fs[xb], self.fs[k]);
        let ev = self.new_event(b, i, &[(xi, fx), (z, fz)]);
        // On a tie the part next to the old point is the larger one.
        let near_big = fx <= fz;
        let g = if near_big { xi + Q * (z - xi) } else { xi + Q * Q * (z - xi) };
        let (s1, s2) = ((s + 1).min(smax), (s + 2).min(smax));
        let small = (z - xi).abs() * (1.0 - Q);
        let mut y = std::mem::take(&mut self.ybuf);
        y.copy_from_slice(&self.opp[b * n..(b + 1) * n]);
        let mut ns = std::mem::take(&mut self.nsbuf);
        ns.copy_from_slice(&self.nsplit[b * n..(b + 1) * n]);
        ns[i] += 1;
        y[i] = g;
        let c1 = self.new_box(xb as u32, &y, &ns, if near_big { s1 } else { s2 }, ev);
        let c2 = self.new_box(k as u32, &y, &ns, if near_big { s2 } else { s1 }, ev);
        let c3 = (z != yi).then(|| {
            y[i] = yi;
            let lev = if !by_gain || (yi - z).abs() > small { s1 } else { s2 };
            self.new_box(k as u32, &y, &ns, lev, ev)
        });
        self.ybuf = y;
        self.nsbuf = ns;
        self.unfile(b);
        self.level[b] = 0;
        self.file(c1);
        self.file(c2);
        if let Some(c3) = c3 {
            self.file(c3);
        }
        Split::Done
    }

    /// Expected gain of Section 4.2 for every coordinate of box `b`:
    /// returns the best coordinate, its gain and its splitting value
    /// (`usize::MAX` if no coordinate can be split).
    #[cfg_attr(feature = "profile", inline(never))]
    fn expected_gain(&mut self, b: usize) -> (usize, f64, f64) {
        let n = self.n;
        let xb = self.base[b] as usize;
        let fx = self.fs[xb];
        // Two points per coordinate from the history, nearest events first.
        let mut cnt = std::mem::take(&mut self.cnt_buf);
        let mut pts = std::mem::take(&mut self.pts_buf);
        cnt.clear();
        cnt.resize(n, 0);
        pts.clear();
        pts.resize(2 * n, (0.0, 0.0));
        let x = &self.xs[xb * n..(xb + 1) * n];
        let mut need = n;
        let mut e = self.born[b];
        while e != crate::NONE && need > 0 {
            let ev = self.events[e as usize];
            let c = ev.coord as usize;
            if cnt[c] < 2 {
                for &(t, ft) in &ev.pts[..ev.len as usize] {
                    if t != x[c] && cnt[c] < 2 {
                        pts[2 * c + cnt[c] as usize] = (t, ft);
                        cnt[c] += 1;
                        if cnt[c] == 2 {
                            need -= 1;
                        }
                    }
                }
            }
            e = ev.parent;
        }
        let (mut best_i, mut best_gain, mut best_z) = (usize::MAX, f64::INFINITY, 0.0);
        for i in 0..n {
            let (lo, hi) = self.interval(b, i);
            if !(hi > lo) {
                continue;
            }
            let (gain, z) = if self.nsplit[b * n + i] == 0 {
                // Case 1: the initialization list's known differences.
                let l = &self.init[i];
                let fmin = l.f.iter().copied().fold(f64::INFINITY, f64::min);
                (fmin - l.f[l.l0], f64::NAN)
            } else {
                let mut c = cnt[i] as usize;
                let mut p = [pts[2 * i], pts[2 * i + 1]];
                // Too little history: fill from the initialization list.
                for (t, ft) in self.init[i].x.iter().copied().zip(self.init[i].f.iter().copied()) {
                    if c < 2 && t != x[i] && (c == 0 || t != p[0].0) {
                        p[c] = (t, ft);
                        c += 1;
                    }
                }
                if c < 2 {
                    continue;
                }
                let (_, a1, a2) = parabola([(x[i], fx), p[0], p[1]]);
                let yi = self.opp[b * n + i];
                let far = subint(x[i], yi);
                let near = x[i] + (far - x[i]) / 10.0;
                let (l, h) = if near <= far { (near, far) } else { (far, near) };
                let e = |t: f64| {
                    let d = t - x[i];
                    d * (a1 + a2 * d)
                };
                let (mut g, mut z) = (e(l), l);
                if e(h) < g {
                    g = e(h);
                    z = h;
                }
                if a2 > 0.0 {
                    let tv = x[i] - a1 / (2.0 * a2);
                    if tv > l && tv < h && e(tv) < g {
                        g = e(tv);
                        z = tv;
                    }
                }
                (g, z)
            };
            if gain < best_gain {
                best_gain = gain;
                best_i = i;
                best_z = z;
            }
        }
        self.cnt_buf = cnt;
        self.pts_buf = pts;
        (best_i, best_gain, best_z)
    }
}
