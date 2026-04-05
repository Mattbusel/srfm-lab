/// Binary action space for the exit-timing agent.
///
/// At each bar the agent chooses one of two actions:
///   - HOLD: remain in the position for at least one more bar
///   - EXIT: close the position at the current bar's close price
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Action {
    /// Stay in the trade; pay the holding opportunity cost.
    Hold = 0,
    /// Close the trade; realize the current P&L.
    Exit = 1,
}

impl Action {
    /// Convert a raw integer index (0 or 1) to an `Action`.
    /// Panics if `idx` is out of range.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Action::Hold,
            1 => Action::Exit,
            _ => panic!("invalid action index: {}", idx),
        }
    }

    /// Return the integer index of this action (used as Q-table column).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Action::Hold => "HOLD",
            Action::Exit => "EXIT",
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// The number of discrete actions available.
pub const NUM_ACTIONS: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_roundtrip() {
        assert_eq!(Action::from_index(0), Action::Hold);
        assert_eq!(Action::from_index(1), Action::Exit);
        assert_eq!(Action::Hold.index(), 0);
        assert_eq!(Action::Exit.index(), 1);
    }

    #[test]
    #[should_panic]
    fn test_invalid_index_panics() {
        Action::from_index(2);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Action::Hold), "HOLD");
        assert_eq!(format!("{}", Action::Exit), "EXIT");
    }
}
