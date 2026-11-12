//! The result of one job, see [`MatrixPhase::for_each_top_k`].
//!
//! [`MatrixPhase::for_each_top_k`]: super::MatrixPhase::for_each_top_k

use super::MatrixTopK;

/// The best columns of each row of a job: what a [`MatrixTopK`] holds for its rows, without
/// the buffers of the matrix call.
///
/// A job gets its own, filled from the buffers its matrix calls ran in, so that those are
/// free again while the job is consumed.
#[derive(Debug)]
pub struct MatrixPicks {
    /// Most picks of a row, the `k` of the job.
    k: usize,
    /// Filled entries per row.
    len: Vec<u32>,
    /// Column positions, `k` per row, best first.
    cols: Vec<u32>,
    /// Scores, parallel to `cols`.
    scores: Vec<f32>,
}

impl MatrixPicks {
    /// `rows` rows without picks, of at most `k` each.
    pub(super) fn new(k: usize, rows: usize) -> Self {
        Self {
            k,
            len: vec![0; rows],
            cols: vec![0; rows * k],
            scores: vec![0.0; rows * k],
        }
    }

    /// Number of rows of the job.
    pub fn rows(&self) -> usize {
        self.len.len()
    }

    /// Row `r`'s column positions and scores, at most the `k` of the job, best first, ties to
    /// the lower column position. NaN scores are never picked.
    pub fn row(&self, r: usize) -> (&[u32], &[f32]) {
        let start = r * self.k;
        let end = start + self.len[r] as usize;
        (&self.cols[start..end], &self.scores[start..end])
    }

    /// Replaces the rows from `at` on with the first `k` picks of each row of `part`, the
    /// result of a matrix call over those rows of the job. `part` may keep more per row: ties
    /// go to the lower column, so its first `k` are the `k` best.
    pub(super) fn fill(&mut self, at: usize, part: &MatrixTopK) {
        for r in 0..part.rows() {
            let (cols, scores) = part.row(r);
            let len = cols.len().min(self.k);
            let start = (at + r) * self.k;
            self.len[at + r] = len as u32;
            self.cols[start..start + len].copy_from_slice(&cols[..len]);
            self.scores[start..start + len].copy_from_slice(&scores[..len]);
        }
    }
}
