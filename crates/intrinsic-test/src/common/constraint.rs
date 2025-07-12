use serde::Deserialize;
use std::ops::Range;

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    Equal(i64),
    Range(Range<i64>),
    Set(Vec<i64>),
}

impl Constraint {
    pub fn to_vector(&self) -> Vec<i64> {
        match self {
            Constraint::Equal(eq) => vec![*eq],
            Constraint::Range(range) => range.clone().collect::<Vec<i64>>(),
            Constraint::Set(values) => values.clone(),
        }
    }
}
