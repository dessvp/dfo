//! Debug driver: Hartman 6 and Shekel 5 with a small cap, printing progress.
use dfo_mcs::{Config, Mcs};
use std::time::Instant;

fn main() {
    let h3 = |x: &[f64]| -> f64 {
        let a = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]];
        let p = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]];
        let al = [1.0, 1.2, 3.0, 3.2];
        -(0..4).map(|j| al[j] * (-(0..3).map(|i| a[j][i] * (x[i] - p[j][i]).powi(2)).sum::<f64>()).exp()).sum::<f64>()
    };
    let cap: usize = std::env::args().nth(1).map_or(2000, |s| s.parse().unwrap());
    let mut cfg = Config::new(3);
    cfg.f_target = -3.86278214782076 * (1.0 - 1e-4);
    let mut m = Mcs::new(&[0.0; 3], &[1.0; 3], cfg);
    let mut f = h3;
    let t = Instant::now();
    let mut last = 0;
    loop {
        let st = m.run(&mut f, 50, cap);
        let e = m.evaluations();
        eprintln!("{st:?} evals {e} best {:.8} t {:.3}s", m.best().unwrap().1, t.elapsed().as_secs_f64());
        if e == last || st != dfo_mcs::Status::Quota {
            break;
        }
        last = e;
    }
}
