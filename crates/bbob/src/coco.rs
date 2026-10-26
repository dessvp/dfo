//! The part of the COCO C API (coco/coco.h) the experiment uses, with a
//! small safe wrapper.

use std::ffi::{CStr, CString, c_char, c_double, c_int};

#[repr(C)]
pub struct CocoSuite {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CocoObserver {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CocoProblem {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn coco_set_log_level(level: *const c_char) -> *const c_char;
    fn coco_suite(name: *const c_char, instance: *const c_char, options: *const c_char) -> *mut CocoSuite;
    fn coco_suite_free(suite: *mut CocoSuite);
    fn coco_observer(name: *const c_char, options: *const c_char) -> *mut CocoObserver;
    fn coco_observer_free(observer: *mut CocoObserver);
    fn coco_suite_get_next_problem(suite: *mut CocoSuite, observer: *mut CocoObserver) -> *mut CocoProblem;
    fn coco_evaluate_function(problem: *mut CocoProblem, x: *const c_double, y: *mut c_double);
    fn coco_problem_get_dimension(problem: *const CocoProblem) -> usize;
    fn coco_problem_get_id(problem: *const CocoProblem) -> *const c_char;
    fn coco_problem_get_evaluations(problem: *const CocoProblem) -> usize;
    fn coco_problem_final_target_hit(problem: *const CocoProblem) -> c_int;
    fn coco_problem_get_best_observed_fvalue1(problem: *const CocoProblem) -> c_double;
    fn coco_problem_get_final_target_fvalue1(problem: *const CocoProblem) -> c_double;
    fn coco_problem_get_smallest_values_of_interest(problem: *const CocoProblem) -> *const c_double;
    fn coco_problem_get_largest_values_of_interest(problem: *const CocoProblem) -> *const c_double;
}

fn cstr(s: &str) -> CString {
    CString::new(s).expect("no interior NUL")
}

pub fn set_log_level(level: &str) {
    let l = cstr(level);
    // SAFETY: COCO copies the level name.
    unsafe { coco_set_log_level(l.as_ptr()) };
}

/// A benchmark suite together with the observer that logs every evaluation.
pub struct Suite {
    suite: *mut CocoSuite,
    observer: *mut CocoObserver,
}

impl Suite {
    pub fn new(name: &str, instance: &str, options: &str, observer: &str, observer_options: &str) -> Self {
        let (n, i, o) = (cstr(name), cstr(instance), cstr(options));
        let (on, oo) = (cstr(observer), cstr(observer_options));
        // SAFETY: valid NUL-terminated strings; COCO copies what it keeps.
        let (suite, observer) = unsafe { (coco_suite(n.as_ptr(), i.as_ptr(), o.as_ptr()), coco_observer(on.as_ptr(), oo.as_ptr())) };
        assert!(!suite.is_null() && !observer.is_null(), "COCO could not create the suite or observer");
        Suite { suite, observer }
    }

    /// The next problem, or `None` when the suite is exhausted. The problem
    /// is owned by the suite and valid until the next call.
    pub fn next_problem(&mut self) -> Option<Problem<'_>> {
        // SAFETY: suite and observer are live.
        let p = unsafe { coco_suite_get_next_problem(self.suite, self.observer) };
        (!p.is_null()).then_some(Problem { p, _suite: std::marker::PhantomData })
    }
}

impl Drop for Suite {
    fn drop(&mut self) {
        // SAFETY: created by coco_observer / coco_suite, freed once.
        unsafe {
            coco_observer_free(self.observer);
            coco_suite_free(self.suite);
        }
    }
}

pub struct Problem<'s> {
    p: *mut CocoProblem,
    _suite: std::marker::PhantomData<&'s mut Suite>,
}

impl Problem<'_> {
    pub fn dimension(&self) -> usize {
        // SAFETY: p is a live problem.
        unsafe { coco_problem_get_dimension(self.p) }
    }

    pub fn id(&self) -> String {
        // SAFETY: COCO returns a NUL-terminated string owned by the problem.
        unsafe { CStr::from_ptr(coco_problem_get_id(self.p)) }.to_string_lossy().into_owned()
    }

    pub fn lower(&self) -> Vec<f64> {
        // SAFETY: an array of `dimension()` doubles owned by the problem.
        unsafe { std::slice::from_raw_parts(coco_problem_get_smallest_values_of_interest(self.p), self.dimension()) }.to_vec()
    }

    pub fn upper(&self) -> Vec<f64> {
        // SAFETY: as for `lower`.
        unsafe { std::slice::from_raw_parts(coco_problem_get_largest_values_of_interest(self.p), self.dimension()) }.to_vec()
    }

    pub fn final_target(&self) -> f64 {
        // SAFETY: p is a live problem.
        unsafe { coco_problem_get_final_target_fvalue1(self.p) }
    }

    pub fn evaluations(&self) -> usize {
        // SAFETY: p is a live problem.
        unsafe { coco_problem_get_evaluations(self.p) }
    }

    pub fn final_target_hit(&self) -> bool {
        // SAFETY: p is a live problem.
        unsafe { coco_problem_final_target_hit(self.p) != 0 }
    }

    pub fn best_observed(&self) -> f64 {
        // SAFETY: p is a live problem.
        unsafe { coco_problem_get_best_observed_fvalue1(self.p) }
    }

    /// Evaluates (and logs) the objective at `x`.
    pub fn eval(&mut self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dimension());
        let mut y = 0.0;
        // SAFETY: x has `dimension()` entries; y receives one value.
        unsafe { coco_evaluate_function(self.p, x.as_ptr(), &mut y) };
        y
    }
}
