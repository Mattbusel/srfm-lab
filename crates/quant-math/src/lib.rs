// quant-math: Core quantitative mathematics library
// Dense, std-only Rust implementation of numerical methods

/// Dense matrix operations, decompositions, and linear system solvers.
pub mod linear_algebra;

/// Descriptive and inferential statistics, correlation, covariance, bootstrap.
pub mod statistics;

/// Probability distributions, random number generation, sampling.
pub mod distributions;

/// Unconstrained and constrained optimization algorithms.
pub mod optimization;

/// Interpolation and approximation: splines, B-splines, rational.
pub mod interpolation;

/// Special mathematical functions: erf, gamma, Bessel, Airy, hypergeometric.
pub mod special_functions;

/// Numerical integration, differentiation, root finding, ODE solvers, FFT.
pub mod numerical;

/// Time series analysis: ARMA, GARCH, smoothing, Kalman filter.
pub mod time_series;

pub use linear_algebra::Matrix;
pub use distributions::Xoshiro256PlusPlus;
pub use numerical::Complex;
