//! Analysis data for a candidate move produced by a search.
//!
//! Corresponds to `cpp/search/analysisdata.h` and `cpp/search/analysisdata.cpp`.
//! The `node` pointer field is omitted because `SearchNode` is not yet ported; it
//! is not used by any of the methods here.

use std::cmp::Ordering;

use kata_game::board::{self, Board, Loc, PASS_LOC, Player};
use kata_game::history::BoardHistory;

use crate::node::SearchNode;

/// Statistics and principal variation for a single candidate move.
#[derive(Debug, Clone, Default)]
pub struct AnalysisData {
    pub move_loc: Loc,
    pub num_visits: i64,
    pub play_selection_value: f64,
    pub lcb: f64,
    pub radius: f64,
    /// Utility, roughly in [-1, 1] or a widened range depending on score utility.
    pub utility: f64,
    pub result_utility: f64,
    pub score_utility: f64,
    pub win_loss_value: f64,
    pub no_result_value: f64,
    pub policy_prior: f64,
    pub score_mean: f64,
    pub score_stdev: f64,
    pub lead: f64,
    /// Effective sample size taking weighting into account.
    pub ess: f64,
    pub weight_factor: f64,
    pub weight_sum: f64,
    pub weight_sq_sum: f64,
    pub utility_sq_avg: f64,
    pub score_mean_sq_avg: f64,
    pub child_visits: i64,
    pub child_weight_sum: f64,
    /// Preference order, 0 is best.
    pub order: i32,
    /// If not `NULL_LOC`, this move is a duplicate reflected from `is_symmetry_of`.
    pub is_symmetry_of: Loc,
    /// The symmetry applied to `is_symmetry_of` to get this move.
    pub symmetry: i32,
    /// Raw pointer to the search tree node that produced this entry, if any.
    /// Used by analysis output to compute per-move ownership.
    pub node: *const SearchNode,
    /// Principal variation.
    pub pv: Vec<Loc>,
    pub pv_visits: Vec<i64>,
    pub pv_edge_visits: Vec<i64>,
}

impl AnalysisData {
    /// Create a new zero-initialized analysis entry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the principal variation contains a pass.
    pub fn pv_contains_pass(&self) -> bool {
        self.pv.contains(&PASS_LOC)
    }

    /// Write the principal variation as GTP coordinates to `out`.
    pub fn write_pv(&self, out: &mut String, board: &Board) {
        for (j, &loc) in self.pv.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&board::location::to_string(loc, board.x_size, board.y_size));
        }
    }

    /// Write the per-PV-node visit counts to `out`.
    pub fn write_pv_visits(&self, out: &mut String) {
        for (j, &visits) in self.pv_visits.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&visits.to_string());
        }
    }

    /// Write the per-PV-edge visit counts to `out`.
    pub fn write_pv_edge_visits(&self, out: &mut String) {
        for (j, &visits) in self.pv_edge_visits.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&visits.to_string());
        }
    }

    /// Return the number of PV moves up to (but not including) the first move
    /// that changes the encore phase.
    pub fn get_pv_len_up_to_phase_end(
        &self,
        initial_board: &Board,
        initial_hist: &BoardHistory,
        initial_pla: Player,
    ) -> usize {
        let mut board = initial_board.clone();
        let mut hist = initial_hist.clone();
        let mut next_pla = initial_pla;
        let mut j = 0;
        while j < self.pv.len() {
            hist.make_board_move_assume_legal(&mut board, self.pv[j], next_pla);
            next_pla = board::get_opp(next_pla);
            if hist.encore_phase != initial_hist.encore_phase {
                break;
            }
            j += 1;
        }
        j
    }

    /// Write the principal variation up to the phase end to `out`.
    pub fn write_pv_up_to_phase_end(
        &self,
        out: &mut String,
        initial_board: &Board,
        initial_hist: &BoardHistory,
        initial_pla: Player,
    ) {
        let mut board = initial_board.clone();
        let mut hist = initial_hist.clone();
        let mut next_pla = initial_pla;
        for (j, &loc) in self.pv.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&board::location::to_string(loc, board.x_size, board.y_size));

            hist.make_board_move_assume_legal(&mut board, loc, next_pla);
            next_pla = board::get_opp(next_pla);
            if hist.encore_phase != initial_hist.encore_phase {
                break;
            }
        }
    }

    /// Write the per-PV-node visit counts up to the phase end to `out`.
    pub fn write_pv_visits_up_to_phase_end(
        &self,
        out: &mut String,
        initial_board: &Board,
        initial_hist: &BoardHistory,
        initial_pla: Player,
    ) {
        assert_eq!(self.pv.len(), self.pv_visits.len());
        let mut board = initial_board.clone();
        let mut hist = initial_hist.clone();
        let mut next_pla = initial_pla;
        for (j, &loc) in self.pv.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&self.pv_visits[j].to_string());

            hist.make_board_move_assume_legal(&mut board, loc, next_pla);
            next_pla = board::get_opp(next_pla);
            if hist.encore_phase != initial_hist.encore_phase {
                break;
            }
        }
    }

    /// Write the per-PV-edge visit counts up to the phase end to `out`.
    pub fn write_pv_edge_visits_up_to_phase_end(
        &self,
        out: &mut String,
        initial_board: &Board,
        initial_hist: &BoardHistory,
        initial_pla: Player,
    ) {
        assert_eq!(self.pv.len(), self.pv_edge_visits.len());
        let mut board = initial_board.clone();
        let mut hist = initial_hist.clone();
        let mut next_pla = initial_pla;
        for (j, &loc) in self.pv.iter().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&self.pv_edge_visits[j].to_string());

            hist.make_board_move_assume_legal(&mut board, loc, next_pla);
            next_pla = board::get_opp(next_pla);
            if hist.encore_phase != initial_hist.encore_phase {
                break;
            }
        }
    }
}

impl PartialEq for AnalysisData {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for AnalysisData {}

impl PartialOrd for AnalysisData {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AnalysisData {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort all 0-visit moves to the end.
        let visits0 = self.num_visits > 0;
        let visits1 = other.num_visits > 0;
        match (visits0, visits1) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }

        // Then by play selection value descending.
        let ord = self
            .play_selection_value
            .partial_cmp(&other.play_selection_value)
            .unwrap_or(Ordering::Equal)
            .reverse();
        if ord != Ordering::Equal {
            return ord;
        }

        // Then by visits descending.
        let ord = self.num_visits.cmp(&other.num_visits).reverse();
        if ord != Ordering::Equal {
            return ord;
        }

        // Then by raw policy prior descending.
        self.policy_prior
            .partial_cmp(&other.policy_prior)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{Board, P_BLACK, location};
    use kata_game::rules::Rules;

    fn empty_analysis() -> AnalysisData {
        AnalysisData::new()
    }

    fn analysis_with_pv(pv: Vec<Loc>) -> AnalysisData {
        AnalysisData {
            move_loc: pv[0],
            num_visits: 10,
            pv,
            pv_visits: vec![1, 2, 3],
            pv_edge_visits: vec![4, 5, 6],
            ..AnalysisData::default()
        }
    }

    #[test]
    fn test_pv_contains_pass() {
        let mut a = empty_analysis();
        a.pv = vec![location::get_loc(3, 3, 9), PASS_LOC];
        assert!(a.pv_contains_pass());

        a.pv = vec![location::get_loc(3, 3, 9)];
        assert!(!a.pv_contains_pass());
    }

    #[test]
    fn test_write_pv() {
        let board = Board::new(9, 9);
        let mut a = empty_analysis();
        a.pv = vec![
            location::get_loc(3, 3, board.x_size),
            location::get_loc(4, 4, board.x_size),
        ];
        let mut s = String::new();
        a.write_pv(&mut s, &board);
        assert_eq!(s, "D6 E5");
    }

    #[test]
    fn test_write_pv_visits() {
        let mut a = empty_analysis();
        a.pv_visits = vec![10, 20, 30];
        let mut s = String::new();
        a.write_pv_visits(&mut s);
        assert_eq!(s, "10 20 30");
    }

    #[test]
    fn test_write_pv_edge_visits() {
        let mut a = empty_analysis();
        a.pv_edge_visits = vec![1, 2, 3];
        let mut s = String::new();
        a.write_pv_edge_visits(&mut s);
        assert_eq!(s, "1 2 3");
    }

    #[test]
    fn test_ordering_zero_visits_last() {
        let mut a0 = empty_analysis();
        a0.num_visits = 0;
        a0.play_selection_value = 100.0;

        let mut a1 = empty_analysis();
        a1.num_visits = 1;
        a1.play_selection_value = 1.0;

        assert!(a1 < a0);
    }

    #[test]
    fn test_ordering_by_play_selection_value() {
        let mut a0 = empty_analysis();
        a0.num_visits = 10;
        a0.play_selection_value = 5.0;

        let mut a1 = empty_analysis();
        a1.num_visits = 10;
        a1.play_selection_value = 10.0;

        assert!(a1 < a0);
    }

    #[test]
    fn test_ordering_by_visits_then_policy() {
        let mut a0 = empty_analysis();
        a0.num_visits = 20;
        a0.play_selection_value = 5.0;
        a0.policy_prior = 0.1;

        let mut a1 = empty_analysis();
        a1.num_visits = 30;
        a1.play_selection_value = 5.0;
        a1.policy_prior = 0.05;

        assert!(a1 < a0);
    }

    #[test]
    fn test_get_pv_len_up_to_phase_end_full_pv() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let a = analysis_with_pv(vec![
            location::get_loc(0, 0, board.x_size),
            location::get_loc(1, 1, board.x_size),
        ]);
        assert_eq!(a.get_pv_len_up_to_phase_end(&board, &hist, P_BLACK), 2);
    }

    #[test]
    fn test_write_pv_up_to_phase_end() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let a = analysis_with_pv(vec![
            location::get_loc(0, 0, board.x_size),
            location::get_loc(1, 1, board.x_size),
        ]);
        let mut s = String::new();
        a.write_pv_up_to_phase_end(&mut s, &board, &hist, P_BLACK);
        assert_eq!(s, "A5 B4");
    }

    #[test]
    fn test_write_pv_visits_up_to_phase_end() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut a = analysis_with_pv(vec![
            location::get_loc(0, 0, board.x_size),
            location::get_loc(1, 1, board.x_size),
        ]);
        a.pv_visits = vec![10, 20];
        let mut s = String::new();
        a.write_pv_visits_up_to_phase_end(&mut s, &board, &hist, P_BLACK);
        assert_eq!(s, "10 20");
    }

    #[test]
    fn test_write_pv_edge_visits_up_to_phase_end() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut a = analysis_with_pv(vec![
            location::get_loc(0, 0, board.x_size),
            location::get_loc(1, 1, board.x_size),
        ]);
        a.pv_edge_visits = vec![3, 7];
        let mut s = String::new();
        a.write_pv_edge_visits_up_to_phase_end(&mut s, &board, &hist, P_BLACK);
        assert_eq!(s, "3 7");
    }
}
