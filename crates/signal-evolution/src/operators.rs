/// Function node operators for the signal expression tree.
///
/// Each operator takes child signal series (Vec<f64>) and produces a new series.
/// All operators are length-preserving (output len == input len).

pub mod crossover;
pub mod mutation;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Operator enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    /// Protected division: returns 1.0 when denominator near zero.
    Div,
    Max,
    Min,
    Abs,
    Sign,
    /// Lag signal by n bars (shift right, fill with 0).
    Lag { n: usize },
    RollingMean { window: usize },
    RollingStd { window: usize },
    /// Clip values to [lo, hi].
    Clip { lo: f64, hi: f64 },
    /// If child[0] > 0 return child[1] else child[2].
    IfPositive,
    /// 1.0 when child[0] crosses above child[1] (prev[0] < prev[1] && cur[0] > cur[1]).
    CrossOver,
    /// 1.0 when child[0] crosses below child[1].
    CrossUnder,
}

impl Operator {
    /// Number of child nodes this operator requires.
    pub fn arity(&self) -> usize {
        match self {
            Operator::Abs | Operator::Sign | Operator::Lag { .. }
            | Operator::RollingMean { .. } | Operator::RollingStd { .. }
            | Operator::Clip { .. } => 1,

            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div
            | Operator::Max | Operator::Min | Operator::CrossOver | Operator::CrossUnder => 2,

            Operator::IfPositive => 3,
        }
    }

    /// Apply operator to pre-evaluated child series.
    pub fn apply(&self, children: &[Vec<f64>]) -> Vec<f64> {
        assert_eq!(
            children.len(),
            self.arity(),
            "Operator {:?} requires {} children, got {}",
            self, self.arity(), children.len()
        );
        let n = children[0].len();
        // Validate all children same length
        for c in children.iter() {
            assert_eq!(c.len(), n, "Child length mismatch in operator {:?}", self);
        }
        match self {
            Operator::Add => elementwise2(children, |a, b| a + b),
            Operator::Sub => elementwise2(children, |a, b| a - b),
            Operator::Mul => elementwise2(children, |a, b| a * b),
            Operator::Div => elementwise2(children, |a, b| {
                if b.abs() < 1e-10 { 1.0 } else { a / b }
            }),
            Operator::Max => elementwise2(children, f64::max),
            Operator::Min => elementwise2(children, f64::min),
            Operator::Abs => children[0].iter().map(|x| x.abs()).collect(),
            Operator::Sign => children[0].iter().map(|x| x.signum()).collect(),

            Operator::Lag { n: lag } => {
                let mut out = vec![0.0f64; n];
                for i in *lag..n {
                    out[i] = children[0][i - lag];
                }
                out
            }

            Operator::RollingMean { window } => {
                crate::data_loader::rolling_mean(&children[0], *window)
            }

            Operator::RollingStd { window } => {
                crate::data_loader::rolling_std(&children[0], *window)
            }

            Operator::Clip { lo, hi } => children[0]
                .iter()
                .map(|x| x.clamp(*lo, *hi))
                .collect(),

            Operator::IfPositive => {
                let cond = &children[0];
                let yes = &children[1];
                let no = &children[2];
                (0..n).map(|i| if cond[i] > 0.0 { yes[i] } else { no[i] }).collect()
            }

            Operator::CrossOver => {
                let mut out = vec![0.0f64; n];
                for i in 1..n {
                    if children[0][i - 1] <= children[1][i - 1]
                        && children[0][i] > children[1][i]
                    {
                        out[i] = 1.0;
                    }
                }
                out
            }

            Operator::CrossUnder => {
                let mut out = vec![0.0f64; n];
                for i in 1..n {
                    if children[0][i - 1] >= children[1][i - 1]
                        && children[0][i] < children[1][i]
                    {
                        out[i] = 1.0;
                    }
                }
                out
            }
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> String {
        match self {
            Operator::Add => "+".to_string(),
            Operator::Sub => "-".to_string(),
            Operator::Mul => "*".to_string(),
            Operator::Div => "/".to_string(),
            Operator::Max => "max".to_string(),
            Operator::Min => "min".to_string(),
            Operator::Abs => "abs".to_string(),
            Operator::Sign => "sign".to_string(),
            Operator::Lag { n } => format!("lag({n})"),
            Operator::RollingMean { window } => format!("rmean({window})"),
            Operator::RollingStd { window } => format!("rstd({window})"),
            Operator::Clip { lo, hi } => format!("clip({lo:.2},{hi:.2})"),
            Operator::IfPositive => "if+".to_string(),
            Operator::CrossOver => "xover".to_string(),
            Operator::CrossUnder => "xunder".to_string(),
        }
    }

    /// All operator variants for random sampling (arity-balanced).
    pub fn all_variants() -> Vec<Operator> {
        vec![
            Operator::Add,
            Operator::Sub,
            Operator::Mul,
            Operator::Div,
            Operator::Max,
            Operator::Min,
            Operator::Abs,
            Operator::Sign,
            Operator::Lag { n: 1 },
            Operator::Lag { n: 5 },
            Operator::RollingMean { window: 10 },
            Operator::RollingMean { window: 20 },
            Operator::RollingStd { window: 10 },
            Operator::RollingStd { window: 20 },
            Operator::Clip { lo: -2.0, hi: 2.0 },
            Operator::Clip { lo: -3.0, hi: 3.0 },
            Operator::IfPositive,
            Operator::CrossOver,
            Operator::CrossUnder,
        ]
    }

    /// Operators that take exactly 1 child.
    pub fn unary_variants() -> Vec<Operator> {
        vec![
            Operator::Abs,
            Operator::Sign,
            Operator::Lag { n: 1 },
            Operator::Lag { n: 5 },
            Operator::RollingMean { window: 10 },
            Operator::RollingMean { window: 20 },
            Operator::RollingStd { window: 10 },
            Operator::RollingStd { window: 20 },
            Operator::Clip { lo: -2.0, hi: 2.0 },
        ]
    }

    /// Operators that take exactly 2 children.
    pub fn binary_variants() -> Vec<Operator> {
        vec![
            Operator::Add,
            Operator::Sub,
            Operator::Mul,
            Operator::Div,
            Operator::Max,
            Operator::Min,
            Operator::CrossOver,
            Operator::CrossUnder,
        ]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn elementwise2(children: &[Vec<f64>], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let n = children[0].len();
    (0..n).map(|i| f(children[0][i], children[1][i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(vals: &[f64]) -> Vec<Vec<f64>> {
        vals.chunks(vals.len() / 2)
            .map(|c| c.to_vec())
            .collect()
    }

    #[test]
    fn add_operator() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = Operator::Add.apply(&[a, b]);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn protected_div_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let result = Operator::Div.apply(&[a, b]);
        assert_eq!(result, vec![1.0, 1.0]);
    }

    #[test]
    fn lag_shifts_correctly() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = Operator::Lag { n: 1 }.apply(&[a]);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn clip_bounds() {
        let a = vec![-5.0, 0.0, 5.0];
        let result = Operator::Clip { lo: -2.0, hi: 2.0 }.apply(&[a]);
        assert_eq!(result, vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn crossover_detects_cross() {
        let a = vec![1.0, 2.0, 3.0]; // fast
        let b = vec![2.0, 2.0, 2.0]; // slow
        // i=1: a[0]=1<=b[0]=2, a[1]=2 not > b[1]=2 → no cross
        // i=2: a[1]=2<=b[1]=2, a[2]=3>b[2]=2 → cross
        let result = Operator::CrossOver.apply(&[a, b]);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn if_positive_selects_correctly() {
        let cond = vec![1.0, -1.0, 0.5];
        let yes = vec![10.0, 10.0, 10.0];
        let no = vec![20.0, 20.0, 20.0];
        let result = Operator::IfPositive.apply(&[cond, yes, no]);
        assert_eq!(result, vec![10.0, 20.0, 10.0]);
    }

    #[test]
    fn arity_matches_apply() {
        let bar = vec![1.0f64; 10];
        for op in Operator::all_variants() {
            let children: Vec<Vec<f64>> = (0..op.arity()).map(|_| bar.clone()).collect();
            let res = op.apply(&children);
            assert_eq!(res.len(), 10, "Operator {:?} wrong output len", op);
        }
    }
}
