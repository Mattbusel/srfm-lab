use thiserror::Error;

#[derive(Debug, Error)]
pub enum FactorError {
    #[error("Insufficient data: need at least {required} observations, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Shape mismatch: {msg}")]
    ShapeMismatch { msg: String },

    #[error("Singular matrix encountered during {operation}")]
    SingularMatrix { operation: String },

    #[error("Optimization failed: {reason}")]
    OptimizationFailed { reason: String },

    #[error("Invalid parameter: {name} = {value}, {constraint}")]
    InvalidParameter {
        name: String,
        value: String,
        constraint: String,
    },

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {field} -- {msg}")]
    Parse { field: String, msg: String },

    #[error("Factor not found: {name}")]
    FactorNotFound { name: String },

    #[error("Numerical error: {msg}")]
    Numerical { msg: String },

    #[error("Attribution error: {msg}")]
    Attribution { msg: String },
}

pub type Result<T> = std::result::Result<T, FactorError>;
