//! End-to-end properties of `minimize` that callers rely on: the budget and
//! the bounds are respected, degenerate inputs work, runs are
//! deterministic, and easy problems are solved.

use dfo_portfolio::{HybridConfig, minimize};
use std::cell::{Cell, RefCell};

/// Runs `minimize` on `f` and checks every evaluation against the bounds;
/// returns (evaluations reported, evaluations made, best value).
fn run(f: impl Fn(&[f64]) -> f64, lo: &[f64], hi: &[f64], max_evals: usize, target: f64) -> (usize, usize, f64) {
    let (count, best) = (Cell::new(0usize), Cell::new(f64::INFINITY));
    let g = |x: &[f64]| {
        assert_eq!(x.len(), lo.len());
        for i in 0..x.len() {
            assert!(x[i] >= lo[i] && x[i] <= hi[i], "x[{i}] = {} outside [{}, {}]", x[i], lo[i], hi[i]);
        }
        count.set(count.get() + 1);
        let v = f(x);
        if v < best.get() {
            best.set(v);
        }
        v
    };
    let used = minimize(g, || best.get() <= target, lo, hi, f64::NEG_INFINITY, max_evals, &HybridConfig::default());
    (used, count.get(), best.get())
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().enumerate().map(|(i, t)| (t - 0.3 - 0.1 * i as f64).powi(2)).sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64 + x.iter().map(|t| (t - 0.7).powi(2) - 10.0 * (2.0 * std::f64::consts::PI * (t - 0.7)).cos()).sum::<f64>()
}

/// An ellipsoid of condition 1e4 with coupled neighbouring coordinates
/// (not separable).
fn ellipsoid(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0;
    for i in 0..n {
        let zi = (x[i] - 1.0) + 0.5 * (x[(i + 1) % n] - 1.0);
        s += 1e4f64.powf(i as f64 / (n - 1).max(1) as f64) * zi * zi;
    }
    s
}

#[test]
fn solves_easy_problems() {
    for n in [1, 2, 5, 10] {
        let (lo, hi) = (vec![-5.0; n], vec![5.0; n]);
        let (_, _, best) = run(sphere, &lo, &hi, 2000 * n, 1e-8);
        assert!(best <= 1e-8, "sphere {n}-D: {best}");
        let (_, _, best) = run(ellipsoid, &lo, &hi, 5000 * n, 1e-8);
        assert!(best <= 1e-8, "ellipsoid {n}-D: {best}");
    }
    // Separable and multimodal: the coordinate-wise solver.
    let (lo, hi) = (vec![-5.0; 3], vec![5.0; 3]);
    let (_, _, best) = run(rastrigin, &lo, &hi, 30_000, 1e-8);
    assert!(best <= 1e-8, "rastrigin 3-D: {best}");
}

#[test]
fn never_exceeds_the_budget() {
    for n in [1, 2, 3, 5, 10, 20] {
        let (lo, hi) = (vec![-5.0; n], vec![5.0; n]);
        for max_evals in [0, 1, 2, 3, 5, 10, 37, 100, 1000, 5000] {
            for f in [sphere as fn(&[f64]) -> f64, rastrigin, ellipsoid] {
                let (used, made, _) = run(f, &lo, &hi, max_evals, f64::NEG_INFINITY);
                assert!(made <= max_evals, "{n}-D, budget {max_evals}: {made} evaluations");
                assert_eq!(used, made, "{n}-D, budget {max_evals}: reported {used}, made {made}");
            }
        }
    }
}

#[test]
fn respects_asymmetric_and_narrow_bounds() {
    let lo = [0.0, -1e-3, 10.0, -100.0];
    let hi = [1.0, 1e-3, 11.0, 300.0];
    let (_, made, best) = run(sphere, &lo, &hi, 20_000, f64::NEG_INFINITY);
    assert!(made > 0 && best.is_finite());
}

#[test]
fn optimum_on_the_boundary() {
    // Minimum at the upper bound in every coordinate.
    let f = |x: &[f64]| x.iter().map(|t| (t - 7.0).powi(2)).sum::<f64>();
    let (lo, hi) = (vec![-5.0; 4], vec![5.0; 4]);
    let (_, _, best) = run(f, &lo, &hi, 20_000, 16.0 * 4.0 + 1e-8);
    assert!(best <= 64.0 + 1e-8, "{best}");
}

#[test]
fn survives_nan_and_infinite_values() {
    // NaN (of either sign) and +inf on part of the box; the minimum is elsewhere.
    let f = |x: &[f64]| {
        if x[0] > 2.0 {
            f64::NAN
        } else if x[0] < -2.0 {
            -f64::NAN
        } else if x[1] > 3.0 {
            f64::INFINITY
        } else {
            sphere(x)
        }
    };
    let (lo, hi) = (vec![-5.0; 3], vec![5.0; 3]);
    let (_, made, best) = run(f, &lo, &hi, 20_000, 1e-8);
    assert!(made <= 20_000);
    assert!(best <= 1e-8, "{best}");
}

#[test]
fn constant_function() {
    let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
    let (used, made, _) = run(|_| 1.0, &lo, &hi, 5000, f64::NEG_INFINITY);
    assert_eq!(used, made);
    assert!(made <= 5000);
}

#[test]
fn deterministic() {
    let trace = |seed: u64| {
        let xs = RefCell::new(Vec::new());
        let cfg = HybridConfig { seed, ..HybridConfig::default() };
        let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
        minimize(
            |x: &[f64]| {
                xs.borrow_mut().extend_from_slice(x);
                rastrigin(x) + ellipsoid(x)
            },
            || false,
            &lo,
            &hi,
            f64::NEG_INFINITY,
            3000,
            &cfg,
        );
        xs.into_inner()
    };
    assert_eq!(trace(1), trace(1));
    assert_ne!(trace(1), trace(2));
}

#[test]
fn stops_when_done() {
    let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
    let count = Cell::new(0usize);
    let used = minimize(
        |x: &[f64]| {
            count.set(count.get() + 1);
            sphere(x)
        },
        || count.get() >= 123,
        &lo,
        &hi,
        f64::NEG_INFINITY,
        100_000,
        &HybridConfig::default(),
    );
    assert_eq!(count.get(), 123);
    assert_eq!(used, 123);
}

#[test]
#[should_panic(expected = "bounds")]
fn rejects_inverted_bounds() {
    minimize(sphere, || false, &[1.0, 0.0], &[0.0, 1.0], f64::NEG_INFINITY, 100, &HybridConfig::default());
}

#[test]
#[should_panic(expected = "bounds")]
fn rejects_mismatched_bounds() {
    minimize(sphere, || false, &[0.0, 0.0], &[1.0], f64::NEG_INFINITY, 100, &HybridConfig::default());
}

#[test]
#[should_panic(expected = "bounds")]
fn rejects_infinite_bounds() {
    minimize(sphere, || false, &[f64::NEG_INFINITY], &[1.0], f64::NEG_INFINITY, 100, &HybridConfig::default());
}

#[test]
#[should_panic(expected = "dimension")]
fn rejects_empty_problem() {
    minimize(sphere, || false, &[], &[], f64::NEG_INFINITY, 100, &HybridConfig::default());
}
