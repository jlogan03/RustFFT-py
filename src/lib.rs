use std::sync::Arc;

use numpy::{Complex32, Complex64};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use rustfft::{Fft, FftDirection, FftPlanner};

#[pyclass]
#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

impl Into<FftDirection> for Direction {
    fn into(self) -> FftDirection {
        match self {
            Direction::Forward => FftDirection::Forward,
            Direction::Inverse => FftDirection::Inverse,
        }
    }
}

#[pyclass]
struct FftPlannerF32 {
    fft: Arc<dyn Fft<f32>>,
}

#[pymethods]
impl FftPlannerF32 {
    #[new]
    fn new(len: usize, direction: Direction) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft(len, direction.into());

        Self { fft }
    }

    fn process<'py>(&self, buffer: Bound<'py, PyArray1<Complex32>>) -> PyResult<()> {
        self.fft.process(buffer.readwrite().as_slice_mut()?);
        Ok(())
    }
}

#[pyclass]
struct FftPlannerF64 {
    fft: Arc<dyn Fft<f64>>,
}

#[pymethods]
impl FftPlannerF64 {
    #[new]
    fn new(len: usize, direction: Direction) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft(len, direction.into());

        Self { fft }
    }

    fn process<'py>(&self, buffer: Bound<'py, PyArray1<Complex64>>) -> PyResult<()> {
        self.fft.process(buffer.readwrite().as_slice_mut()?);
        Ok(())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::{Direction, FftPlannerF32, FftPlannerF64};
}
