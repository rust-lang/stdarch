use serde::Deserialize;
use std::ops::Range;

/// Describes the values to test for a const generic parameter.
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    /// Test a single value.
    Equal(i64),
    /// Test a range of values, e.g. `0..16`.
    Range(Range<i64>),
}

impl Constraint {
    pub fn to_range(&self) -> Range<i64> {
        match self {
            Constraint::Equal(eq) => *eq..*eq + 1,
            Constraint::Range(range) => range.clone(),
        }
    }
}
