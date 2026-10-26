//! COCO/BBOB benchmark driver for the open solvers.
//!
//! `dfo-bbob --engine portfolio|mcs|cmaes|lqcma --dims 2,3,5 --functions 1-24
//!  --instances "instances: 101-115" --budget 100000 --out NAME --name ALG`
//!
//! Every problem gets `budget * n` evaluations; runs stop at COCO's final
//! target (f_opt + 1e-8).

mod coco;

use std::time::Instant;

/// Flags this binary understands; anything else is rejected.
const FLAGS: &[&str] = &[
    "--adaptive",
    "--best-boost",
    "--budget",
    "--de-start",
    "--de-weight",
    "--dims",
    "--engine",
    "--full-max",
    "--funnel-first",
    "--funnel-start",
    "--funnel-weight",
    "--functions",
    "--instances",
    "--mcs-evals",
    "--mcs-regime",
    "--mcs-rho",
    "--name",
    "--out",
    "--qn",
    "--qn-h0",
    "--qn-restart",
    "--qn-weight",
    "--qn-warm",
    "--seed",
    "--sep-first",
    "--sep-grid",
    "--separable",
    "--sigma-decay",
    "--sigma-first",
    "--sigma-restart",
    "--small-regime",
    "--tabu",
    "--warm-cap",
    "--warm-cond",
    "--weights",
];

/// The value after `key`, or `default`.
fn arg(args: &[String], key: &str, default: &str) -> String {
    debug_assert!(FLAGS.contains(&key));
    match args.iter().position(|a| a == key) {
        None => default.to_string(),
        Some(i) => match args.get(i + 1) {
            Some(v) if !v.starts_with("--") => v.clone(),
            _ => panic!("{key} needs a value"),
        },
    }
}

/// Seed of one problem: a hash of its id (function, instance, dimension),
/// so it does not depend on which other problems are run, mixed with
/// `--seed`.
fn problem_seed(id: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate().skip(1) {
        let is_value = i > 1 && FLAGS.contains(&args[i - 1].as_str());
        if !is_value && !FLAGS.contains(&a.as_str()) {
            panic!("unknown argument {a}");
        }
    }
    let engine = arg(&args, "--engine", "portfolio");
    let budget: f64 = arg(&args, "--budget", "100000").parse().expect("--budget");
    let dims = arg(&args, "--dims", "2,3,5,10,20,40");
    let functions = arg(&args, "--functions", "1-24");
    // COCO wants "instances: 101-115" and silently ignores anything else;
    // accept a bare list too.
    let instances = match arg(&args, "--instances", "") {
        i if i.is_empty() || i.contains(':') => i,
        i => format!("instances: {i}"),
    };
    let out = arg(&args, "--out", "dfo");
    let name = arg(&args, "--name", "dfo-portfolio");
    let full_max: usize = arg(&args, "--full-max", "40").parse().expect("--full-max");
    dfo_cmaes::lq::FULL_MAX_N.store(full_max, std::sync::atomic::Ordering::Relaxed);

    coco::set_log_level("warning");
    let suite_options = format!("dimensions: {dims} function_indices: {functions}");
    let info = format!("dfo {engine} (open implementation)");
    let observer_options = format!("result_folder: {out} algorithm_name: {name} algorithm_info: \"{info}\"");
    let mut suite = coco::Suite::new("bbob", &instances, &suite_options, "bbob", &observer_options);

    let start = Instant::now();
    let (mut solved, mut total) = (0, 0);
    while let Some(mut problem) = suite.next_problem() {
        let n = problem.dimension();
        let problem_id = problem.id();
        let (u, v) = (problem.lower(), problem.upper());
        let ftarget = problem.final_target();
        let max_evals = (budget * n as f64) as usize;
        let t = Instant::now();
        let p = std::cell::RefCell::new(&mut problem);
        let f = |y: &[f64]| p.borrow_mut().eval(y);
        let done = || p.borrow().final_target_hit();
        match engine.as_str() {
            "portfolio" => {
                let cfg = dfo_portfolio::HybridConfig {
                    mcs_first: match arg(&args, "--mcs-evals", "auto").as_str() {
                        "auto" => None,
                        v => Some(v.parse().expect("--mcs-evals")),
                    },
                    sigma_frac: arg(&args, "--sigma-first", "0.1").parse().expect("--sigma-first"),
                    mcs_regime: arg(&args, "--mcs-regime", "yes") == "yes",
                    small_regime: arg(&args, "--small-regime", "yes") == "yes",
                    separable: arg(&args, "--separable", "yes") == "yes",
                    weights: match arg(&args, "--weights", "auto").as_str() {
                        "auto" => None,
                        w => {
                            let w: Vec<f64> = w.split(',').map(|t| t.parse().expect("--weights")).collect();
                            Some([w[0], w[1], w[2]])
                        }
                    },
                    adaptive: arg(&args, "--adaptive", "no") == "yes",
                    mcs_rho_min: arg(&args, "--mcs-rho", "0.02").parse().expect("--mcs-rho"),
                    qn: match arg(&args, "--qn", "auto").as_str() {
                        "auto" => None,
                        q => Some(q.parse().expect("--qn")),
                    },
                    sigma_decay: arg(&args, "--sigma-decay", "1").parse().expect("--sigma-decay"),
                    best_boost: arg(&args, "--best-boost", "no") == "yes",
                    qn_h0: match arg(&args, "--qn-h0", "auto").as_str() {
                        "auto" => None,
                        h => Some(h.parse().expect("--qn-h0")),
                    },
                    qn_warm: arg(&args, "--qn-warm", "yes") == "yes",
                    sigma_restart: match arg(&args, "--sigma-restart", "auto").as_str() {
                        "auto" => None,
                        r => Some(r.parse().expect("--sigma-restart")),
                    },
                    sep_first: match arg(&args, "--sep-first", "auto").as_str() {
                        "auto" => None,
                        v => Some(v == "yes"),
                    },
                    sep_grid: arg(&args, "--sep-grid", "81").parse().expect("--sep-grid"),
                    warm_cap: arg(&args, "--warm-cap", "200").parse().expect("--warm-cap"),
                    qn_weight: arg(&args, "--qn-weight", "0").parse().expect("--qn-weight"),
                    qn_restart: arg(&args, "--qn-restart", "50").parse().expect("--qn-restart"),
                    warm_cond: arg(&args, "--warm-cond", "100").parse().expect("--warm-cond"),
                    funnel_first: match arg(&args, "--funnel-first", "auto").as_str() {
                        "auto" => None,
                        w => Some(w.parse().expect("--funnel-first")),
                    },
                    funnel_start: arg(&args, "--funnel-start", "10000").parse().expect("--funnel-start"),
                    funnel_weight: match arg(&args, "--funnel-weight", "auto").as_str() {
                        "auto" => None,
                        w => Some(w.parse().expect("--funnel-weight")),
                    },
                    de_start: arg(&args, "--de-start", "10000").parse().expect("--de-start"),
                    de_weight: match arg(&args, "--de-weight", "auto").as_str() {
                        "auto" => None,
                        w => Some(w.parse().expect("--de-weight")),
                    },
                    tabu: arg(&args, "--tabu", "0").parse().expect("--tabu"),
                    seed: problem_seed(&problem_id, arg(&args, "--seed", "0").parse::<u64>().expect("--seed")),
                };
                dfo_portfolio::minimize(f, done, &u, &v, ftarget, max_evals, &cfg);
            }
            "mcs" => {
                let mut cfg = dfo_mcs::Config::new(n);
                cfg.f_target = ftarget;
                let mut m = dfo_mcs::Mcs::new(&u, &v, cfg);
                let mut f = f;
                m.run(&mut f, max_evals, max_evals);
            }
            "cmaes" => {
                let (lo, hi) = (vec![-4.0; n], vec![4.0; n]);
                let mut f = f;
                dfo_cmaes::cmaes::ipop(&mut f, done, &lo, &hi, max_evals, problem_seed(&problem_id, 0));
            }
            "lqcma" => {
                let (lo, hi) = (vec![-4.0; n], vec![4.0; n]);
                let mut ar = dfo_cmaes::lq::Archive::default();
                let mut f = f;
                dfo_cmaes::lq::ipop_lq(&mut f, done, &lo, &hi, max_evals, problem_seed(&problem_id, 0), &mut ar, None);
            }
            e => panic!("unknown --engine {e}"),
        }
        total += 1;
        let hit = problem.final_target_hit();
        solved += usize::from(hit);
        println!(
            "{:28} evals {:8} best-ft {:10.3e} {} {:7.2}s",
            problem.id(),
            problem.evaluations(),
            problem.best_observed() - ftarget,
            if hit { "solved" } else { "      " },
            t.elapsed().as_secs_f64()
        );
    }
    println!("{solved}/{total} problems reached the final target in {:.1}s", start.elapsed().as_secs_f64());
}
