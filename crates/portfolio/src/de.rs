//! L-SHADE: success-history based adaptive differential evolution with
//! linear population size reduction, implemented from R. Tanabe and A. S.
//! Fukunaga, "Improving the search performance of SHADE using linear
//! population size reduction", IEEE CEC 2014, with the parameters given
//! there (N_init = 18 n, N_min = 4, H = 6, archive rate 2.6, p = 0.11).
//! As in SHADE (Tanabe & Fukunaga, CEC 2013) the p-best set has at least
//! two members, and as in JADE (Zhang & Sanderson, IEEE TEC 13, 2009) a
//! component outside the box is set halfway between the parent and the
//! bound.
//!
//! The archive-based DE variants lead the COCO/BBOB archive at large
//! budgets in two and three dimensions, on functions such as f24 (Lunacek
//! bi-Rastrigin) that the CMA-ES and MCS regimes rarely solve.

use dfo_cmaes::cmaes::Rng;

/// One L-SHADE run over the box `[u, v]` with `max_evals` evaluations (the
/// population shrinks linearly over them), until `done()`. Returns the
/// evaluations used, the best point and its value.
pub fn lshade<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    done: &mut dyn FnMut() -> bool,
    u: &[f64],
    v: &[f64],
    max_evals: usize,
    rng: &mut Rng,
) -> (usize, Vec<f64>, f64) {
    const H: usize = 6;
    const ARC_RATE: f64 = 2.6;
    const P_BEST: f64 = 0.11;
    const N_MIN: usize = 4;
    let n = u.len();
    let n_init = (18 * n).max(N_MIN + 1);
    // CR = None is the paper's terminal value (crossover off from then on).
    let mut m_cr: [Option<f64>; H] = [Some(0.5); H];
    let mut m_f = [0.5f64; H];
    let mut k_mem = 0;
    let mut used = 0;
    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(n_init);
    let mut fit: Vec<f64> = Vec::with_capacity(n_init);
    for _ in 0..n_init {
        if used >= max_evals || done() {
            break;
        }
        let x: Vec<f64> = (0..n).map(|j| u[j] + rng.uniform() * (v[j] - u[j])).collect();
        fit.push(f(&x));
        pop.push(x);
        used += 1;
    }
    let mut archive: Vec<Vec<f64>> = Vec::new();
    let mut trial = vec![0.0; n];
    let mut order: Vec<usize> = Vec::new();
    let (mut s_cr, mut s_f, mut s_w) = (Vec::new(), Vec::new(), Vec::new());
    while used < max_evals && !done() && pop.len() >= N_MIN {
        let np = pop.len();
        order.clear();
        order.extend(0..np);
        order.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
        let n_pbest = ((P_BEST * np as f64).round() as usize).max(2).min(np);
        s_cr.clear();
        s_f.clear();
        s_w.clear();
        let mut next_pop = pop.clone();
        let mut next_fit = fit.clone();
        for i in 0..np {
            if used >= max_evals || done() {
                break;
            }
            let r = (rng.uniform() * H as f64) as usize % H;
            let cr = match m_cr[r] {
                None => 0.0,
                Some(m) => (m + 0.1 * rng.normal()).clamp(0.0, 1.0),
            };
            let fs = loop {
                let c = m_f[r] + 0.1 * (std::f64::consts::PI * (rng.uniform() - 0.5)).tan();
                if c > 0.0 {
                    break c.min(1.0);
                }
            };
            let pbest = order[(rng.uniform() * n_pbest as f64) as usize % n_pbest];
            let r1 = loop {
                let c = (rng.uniform() * np as f64) as usize % np;
                if c != i {
                    break c;
                }
            };
            let total = np + archive.len();
            let r2 = loop {
                let c = (rng.uniform() * total as f64) as usize % total;
                if c != i && c != r1 {
                    break c;
                }
            };
            let x2 = if r2 < np { &pop[r2] } else { &archive[r2 - np] };
            let jrand = (rng.uniform() * n as f64) as usize % n;
            let xi = &pop[i];
            for j in 0..n {
                trial[j] = if j == jrand || rng.uniform() < cr {
                    let t = xi[j] + fs * (pop[pbest][j] - xi[j]) + fs * (pop[r1][j] - x2[j]);
                    // Out of the box: halfway between the parent and the bound.
                    if t < u[j] {
                        0.5 * (u[j] + xi[j])
                    } else if t > v[j] {
                        0.5 * (v[j] + xi[j])
                    } else {
                        t
                    }
                } else {
                    xi[j]
                };
            }
            let ft = f(&trial);
            used += 1;
            if ft <= fit[i] {
                if ft < fit[i] {
                    archive.push(pop[i].clone());
                    s_cr.push(cr);
                    s_f.push(fs);
                    s_w.push(fit[i] - ft);
                }
                next_pop[i].copy_from_slice(&trial);
                next_fit[i] = ft;
            }
        }
        pop = next_pop;
        fit = next_fit;
        // Success-history update with weighted Lehmer means.
        if !s_cr.is_empty() {
            let wsum: f64 = s_w.iter().sum();
            if wsum > 0.0 && wsum.is_finite() {
                let lehmer = |s: &[f64]| {
                    let num: f64 = s.iter().zip(&s_w).map(|(x, w)| w * x * x).sum();
                    let den: f64 = s.iter().zip(&s_w).map(|(x, w)| w * x).sum();
                    if den > 0.0 { num / den } else { 0.0 }
                };
                m_cr[k_mem] = match m_cr[k_mem] {
                    Some(_) if s_cr.iter().any(|&c| c > 0.0) => Some(lehmer(&s_cr)),
                    _ => None,
                };
                m_f[k_mem] = lehmer(&s_f);
                k_mem = (k_mem + 1) % H;
            }
        }
        // Linear population size reduction: drop the worst.
        let target = (((N_MIN as f64 - n_init as f64) / max_evals as f64) * used as f64 + n_init as f64).round().max(N_MIN as f64) as usize;
        if target < pop.len() {
            order.clear();
            order.extend(0..pop.len());
            order.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
            order.truncate(target);
            pop = order.iter().map(|&k| pop[k].clone()).collect();
            fit = order.iter().map(|&k| fit[k]).collect();
        }
        let cap = (ARC_RATE * pop.len() as f64).round() as usize;
        while archive.len() > cap {
            let k = (rng.uniform() * archive.len() as f64) as usize % archive.len();
            archive.swap_remove(k);
        }
    }
    let b = (0..fit.len()).min_by(|&a, &b| fit[a].total_cmp(&fit[b])).unwrap_or(0);
    (used, pop.get(b).cloned().unwrap_or_else(|| u.to_vec()), fit.get(b).copied().unwrap_or(f64::INFINITY))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_rastrigin_2d() {
        let mut f = |x: &[f64]| 20.0 + x.iter().map(|t| (t - 1.3) * (t - 1.3) - 10.0 * (2.0 * std::f64::consts::PI * (t - 1.3)).cos()).sum::<f64>();
        let mut rng = Rng::new(3);
        let (used, _, fb) = lshade(&mut f, &mut || false, &[-5.0; 2], &[5.0; 2], 20_000, &mut rng);
        assert!(used <= 20_000 && fb < 1e-8, "f = {fb}");
    }
}
