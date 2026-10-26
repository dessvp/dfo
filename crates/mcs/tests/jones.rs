//! The nine test problems of Jones et al. (1993) and Table I of the MCS
//! paper (standard definitions from Dixon & Szegő, 1978), solved to the
//! paper's criterion: relative error below 1e-4 in the global minimum.
//! The global minima f_glob are the published values for these standard
//! test functions (Dixon & Szegő 1978; Jones et al. 1993).

use dfo_mcs::{Config, Mcs, Status};

struct Problem {
    name: &'static str,
    u: Vec<f64>,
    v: Vec<f64>,
    fglob: f64,
    f: fn(&[f64]) -> f64,
}

const SHEKEL_A: [[f64; 4]; 10] = [
    [4.0, 4.0, 4.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0, 8.0],
    [6.0, 6.0, 6.0, 6.0],
    [3.0, 7.0, 3.0, 7.0],
    [2.0, 9.0, 2.0, 9.0],
    [5.0, 5.0, 3.0, 3.0],
    [8.0, 1.0, 8.0, 1.0],
    [6.0, 2.0, 6.0, 2.0],
    [7.0, 3.6, 7.0, 3.6],
];
const SHEKEL_C: [f64; 10] = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5];

fn shekel(x: &[f64], m: usize) -> f64 {
    -(0..m).map(|j| 1.0 / (SHEKEL_C[j] + (0..4).map(|i| (x[i] - SHEKEL_A[j][i]).powi(2)).sum::<f64>())).sum::<f64>()
}

const H_ALPHA: [f64; 4] = [1.0, 1.2, 3.0, 3.2];
const H3_A: [[f64; 3]; 4] = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]];
const H3_P: [[f64; 3]; 4] = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]];
const H6_A: [[f64; 6]; 4] =
    [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0], [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]];
const H6_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

fn hartman3(x: &[f64]) -> f64 {
    -(0..4).map(|j| H_ALPHA[j] * (-(0..3).map(|i| H3_A[j][i] * (x[i] - H3_P[j][i]).powi(2)).sum::<f64>()).exp()).sum::<f64>()
}
fn hartman6(x: &[f64]) -> f64 {
    -(0..4).map(|j| H_ALPHA[j] * (-(0..6).map(|i| H6_A[j][i] * (x[i] - H6_P[j][i]).powi(2)).sum::<f64>()).exp()).sum::<f64>()
}
fn goldstein_price(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (1.0 + (a + b + 1.0).powi(2) * (19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b + 6.0 * a * b + 3.0 * b * b))
        * (30.0 + (2.0 * a - 3.0 * b).powi(2) * (18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b + 27.0 * b * b))
}
fn branin(x: &[f64]) -> f64 {
    let pi = std::f64::consts::PI;
    let (a, b, c) = (1.0, 5.1 / (4.0 * pi * pi), 5.0 / pi);
    a * (x[1] - b * x[0] * x[0] + c * x[0] - 6.0).powi(2) + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * x[0].cos() + 10.0
}
fn camel(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (4.0 - 2.1 * a * a + a.powi(4) / 3.0) * a * a + a * b + (-4.0 + 4.0 * b * b) * b * b
}
fn shubert(x: &[f64]) -> f64 {
    let s = |t: f64| (1..=5).map(|i| i as f64 * ((i as f64 + 1.0) * t + i as f64).cos()).sum::<f64>();
    s(x[0]) * s(x[1])
}

fn problems() -> Vec<Problem> {
    vec![
        Problem { name: "S5", u: vec![0.0; 4], v: vec![10.0; 4], fglob: -10.1531996790582, f: |x| shekel(x, 5) },
        Problem { name: "S7", u: vec![0.0; 4], v: vec![10.0; 4], fglob: -10.4029405668187, f: |x| shekel(x, 7) },
        Problem { name: "S10", u: vec![0.0; 4], v: vec![10.0; 4], fglob: -10.5364098166920, f: |x| shekel(x, 10) },
        Problem { name: "H3", u: vec![0.0; 3], v: vec![1.0; 3], fglob: -3.86278214782076, f: hartman3 },
        Problem { name: "H6", u: vec![0.0; 6], v: vec![1.0; 6], fglob: -3.32236801141551, f: hartman6 },
        Problem { name: "GP", u: vec![-2.0; 2], v: vec![2.0; 2], fglob: 3.0, f: goldstein_price },
        Problem { name: "BR", u: vec![-5.0, 0.0], v: vec![10.0, 15.0], fglob: 0.397887357729739, f: branin },
        Problem { name: "C6", u: vec![-3.0, -2.0], v: vec![3.0, 2.0], fglob: -1.03162845348988, f: camel },
        Problem { name: "SHU", u: vec![-10.0; 2], v: vec![10.0; 2], fglob: -186.730908831024, f: shubert },
    ]
}

/// Evaluations to relative error 1e-4, as in the paper's Table II.
fn solve(p: &Problem) -> (usize, f64) {
    let n = p.u.len();
    let mut cfg = Config::new(n);
    cfg.f_target = p.fglob + 1e-4 * p.fglob.abs();
    let mut mcs = Mcs::new(&p.u, &p.v, cfg);
    let mut f = p.f;
    let st = mcs.run(&mut f, 20_000, 20_000);
    let fb = mcs.best().unwrap().1;
    assert_eq!(st, Status::Target, "{}: best {fb} after {} evaluations", p.name, mcs.evaluations());
    eprintln!("     phases init/split/screen/local {:?}, basket {}", mcs.stats, mcs.basket_len());
    (mcs.evaluations(), fb)
}

#[test]
fn jones_set_reaches_table_ii_criterion() {
    // Table II, row "MCS (e)", for comparison.
    let paper = [83, 129, 103, 79, 111, 81, 41, 42, 69];
    let (mut ours, mut theirs) = (0, 0);
    for (p, &t2) in problems().iter().zip(&paper) {
        let (k, fb) = solve(p);
        eprintln!("{:4} evaluations {k:5}  (paper {t2:4})  f = {fb:.10}", p.name);
        assert!(k <= 2000, "{} took {k} evaluations", p.name);
        ours += k;
        theirs += t2;
    }
    eprintln!("total {ours} (paper {theirs})");
}

#[test]
fn resumable_runs_match_one_long_run() {
    let p = &problems()[4];
    let n = p.u.len();
    let cfg = Config::new(n);
    let mut a = Mcs::new(&p.u, &p.v, cfg.clone());
    let mut fa = p.f;
    a.run(&mut fa, 600, 600);
    let mut b = Mcs::new(&p.u, &p.v, cfg);
    let mut fb = p.f;
    loop {
        let before = b.evaluations();
        if b.run(&mut fb, 7, 600) != Status::Quota || b.evaluations() == before {
            break;
        }
    }
    assert_eq!(a.archive().1, b.archive().1);
}
