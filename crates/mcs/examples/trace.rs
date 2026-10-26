//! Prints the first evaluations on a shifted sphere.
use dfo_mcs::{Config, Mcs};
fn main() {
    let c = [1.3, -2.1];
    let mut f = |x: &[f64]| (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
    let mut m = Mcs::new(&[-5.0; 2], &[5.0; 2], Config::new(2));
    m.run(&mut f, 30, 30);
    let (xs, fs) = m.archive();
    for (k, fk) in fs.iter().enumerate() {
        println!("{k:3} x = ({:8.4}, {:8.4})  f = {fk:.3e}  phase-stats {:?}", xs[2 * k], xs[2 * k + 1], m.stats);
    }
}
