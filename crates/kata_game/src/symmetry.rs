//! Board symmetry helpers.
//!
//! Mirrors `cpp/neuralnet/nninputs.cpp` `SymmetryHelpers`.

use crate::board::{Board, C_EMPTY, Loc, MAX_ARR_SIZE, NULL_LOC, P_BLACK, PASS_LOC, location};
use crate::history::BoardHistory;

pub const NUM_SYMMETRIES: i32 = 8;
pub const NUM_SYMMETRIES_WITHOUT_TRANSPOSE: i32 = 4;

/// Return true if the symmetry transposes x and y.
pub fn is_transpose(symmetry: i32) -> bool {
    (symmetry & 0x4) != 0
}

/// Return true if the symmetry flips x.
pub fn is_flip_x(symmetry: i32) -> bool {
    (symmetry & 0x2) != 0
}

/// Return true if the symmetry flips y.
pub fn is_flip_y(symmetry: i32) -> bool {
    (symmetry & 0x1) != 0
}

/// Apply a symmetry to integer coordinates.
pub fn get_sym_loc_xy(x: i32, y: i32, x_size: i32, y_size: i32, symmetry: i32) -> (i32, i32) {
    let transpose = is_transpose(symmetry);
    let flip_x = is_flip_x(symmetry);
    let flip_y = is_flip_y(symmetry);

    let mut x = if flip_x { x_size - x - 1 } else { x };
    let mut y = if flip_y { y_size - y - 1 } else { y };
    if transpose {
        std::mem::swap(&mut x, &mut y);
    }
    (x, y)
}

/// Apply a symmetry to a location on a board.
pub fn get_sym_loc(loc: Loc, board: &Board, symmetry: i32) -> Loc {
    if loc == NULL_LOC || loc == PASS_LOC {
        return loc;
    }
    let x = location::get_x(loc, board.x_size);
    let y = location::get_y(loc, board.x_size);
    let (sx, sy) = get_sym_loc_xy(x, y, board.x_size, board.y_size, symmetry);
    location::get_loc(
        sx,
        sy,
        if is_transpose(symmetry) {
            board.y_size
        } else {
            board.x_size
        },
    )
}

/// Apply a symmetry to a location given explicit board dimensions.
pub fn get_sym_loc_with_size(loc: Loc, x_size: i32, y_size: i32, symmetry: i32) -> Loc {
    if loc == NULL_LOC || loc == PASS_LOC {
        return loc;
    }
    let x = location::get_x(loc, x_size);
    let y = location::get_y(loc, x_size);
    let (sx, sy) = get_sym_loc_xy(x, y, x_size, y_size, symmetry);
    location::get_loc(
        sx,
        sy,
        if is_transpose(symmetry) {
            y_size
        } else {
            x_size
        },
    )
}

/// Return the symmetry that undoes the given symmetry.
pub fn invert(symmetry: i32) -> i32 {
    if symmetry == 5 {
        return 6;
    }
    if symmetry == 6 {
        return 5;
    }
    symmetry
}

/// Compose two symmetries: first apply `first`, then `next`.
pub fn compose(first: i32, next: i32) -> i32 {
    let mut next = next;
    if is_transpose(first) {
        next = (next & 0x4) | ((next & 0x2) >> 1) | ((next & 0x1) << 1);
    }
    first ^ next
}

/// Compose three symmetries.
pub fn compose_three(first: i32, next: i32, next_next: i32) -> i32 {
    compose(compose(first, next), next_next)
}

/// Copy a batch of spatial input features with the given symmetry.
///
/// `src` and `dst` are flat buffers of length `n_size * h_size * w_size * c_size`.
/// If `use_nhwc` is true the layout is `[N, H, W, C]`, otherwise `[N, C, H, W]`.
#[allow(clippy::too_many_arguments)]
pub fn copy_inputs_with_symmetry(
    src: &[f32],
    dst: &mut [f32],
    n_size: i32,
    h_size: i32,
    w_size: i32,
    c_size: i32,
    use_nhwc: bool,
    symmetry: i32,
) {
    copy_with_symmetry(
        src, dst, n_size, h_size, w_size, c_size, use_nhwc, symmetry, false,
    );
}

/// Copy a batch of scalar outputs (e.g. policy logits) with the given symmetry.
///
/// `src` and `dst` are flat buffers of length `n_size * h_size * w_size`.
pub fn copy_outputs_with_symmetry(
    src: &[f32],
    dst: &mut [f32],
    n_size: i32,
    h_size: i32,
    w_size: i32,
    symmetry: i32,
) {
    copy_with_symmetry(src, dst, n_size, h_size, w_size, 1, false, symmetry, true);
}

#[allow(clippy::too_many_arguments)]
fn copy_with_symmetry(
    src: &[f32],
    dst: &mut [f32],
    n_size: i32,
    h_size: i32,
    w_size: i32,
    c_size: i32,
    use_nhwc: bool,
    symmetry: i32,
    reverse: bool,
) {
    let transpose = is_transpose(symmetry) && h_size == w_size;
    let mut flip_x = is_flip_x(symmetry);
    let mut flip_y = is_flip_y(symmetry);
    if transpose && !reverse {
        std::mem::swap(&mut flip_x, &mut flip_y);
    }

    if use_nhwc {
        let n_stride = (h_size * w_size * c_size) as usize;
        let h_stride = (w_size * c_size) as usize;
        let w_stride = c_size as usize;
        let mut h_base_new = 0isize;
        let mut h_stride_new = h_stride as isize;
        let mut w_base_new = 0isize;
        let mut w_stride_new = w_stride as isize;

        if flip_y {
            h_base_new = ((h_size - 1) * h_stride_new as i32) as isize;
            h_stride_new = -h_stride_new;
        }
        if flip_x {
            w_base_new = ((w_size - 1) * w_stride_new as i32) as isize;
            w_stride_new = -w_stride_new;
        }
        if transpose {
            std::mem::swap(&mut h_stride_new, &mut w_stride_new);
        }

        for n in 0..n_size as usize {
            for h in 0..h_size as usize {
                let nh_old = n * n_stride + h * h_stride;
                let nh_new = (n * n_stride) as isize + h_base_new + (h as isize) * h_stride_new;
                for w in 0..w_size as usize {
                    let nhw_old = nh_old + w * w_stride;
                    let nhw_new = nh_new + w_base_new + (w as isize) * w_stride_new;
                    for c in 0..c_size as usize {
                        dst[(nhw_new as usize) + c] = src[nhw_old + c];
                    }
                }
            }
        }
    } else {
        let nc_size = (n_size * c_size) as usize;
        let nc_stride = (h_size * w_size) as usize;
        let h_stride = w_size as usize;
        let w_stride = 1usize;
        let mut h_base_new = 0isize;
        let mut h_stride_new = h_stride as isize;
        let mut w_base_new = 0isize;
        let mut w_stride_new = w_stride as isize;

        if flip_y {
            h_base_new = ((h_size - 1) * h_stride_new as i32) as isize;
            h_stride_new = -h_stride_new;
        }
        if flip_x {
            w_base_new = ((w_size - 1) * w_stride_new as i32) as isize;
            w_stride_new = -w_stride_new;
        }
        if transpose {
            std::mem::swap(&mut h_stride_new, &mut w_stride_new);
        }

        for nc in 0..nc_size {
            for h in 0..h_size as usize {
                let nch_old = nc * nc_stride + h * h_stride;
                let nch_new = (nc * nc_stride) as isize + h_base_new + (h as isize) * h_stride_new;
                for w in 0..w_size as usize {
                    let nchw_old = nch_old + w * w_stride;
                    let nchw_new = nch_new + w_base_new + (w as isize) * w_stride_new;
                    dst[nchw_new as usize] = src[nchw_old];
                }
            }
        }
    }
}

/// Build the board that results from applying `symmetry` to `board`.
pub fn get_sym_board(board: &Board, symmetry: i32) -> Board {
    let transpose = is_transpose(symmetry);
    let flip_x = is_flip_x(symmetry);
    let flip_y = is_flip_y(symmetry);

    let mut sym_board = Board::new(
        if transpose {
            board.y_size
        } else {
            board.x_size
        },
        if transpose {
            board.x_size
        } else {
            board.y_size
        },
    );

    let mut sym_ko_loc = NULL_LOC;
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            let mut sym_x = if flip_x { board.x_size - x - 1 } else { x };
            let mut sym_y = if flip_y { board.y_size - y - 1 } else { y };
            if transpose {
                std::mem::swap(&mut sym_x, &mut sym_y);
            }
            let sym_loc = location::get_loc(sym_x, sym_y, sym_board.x_size);
            let stone = board.colors[loc as usize];
            if stone != C_EMPTY {
                let success = sym_board.set_stone_fail_if_no_libs(sym_loc, stone);
                assert!(success, "symmetric stone placement should not fail");
            }
            if loc == board.ko_loc {
                sym_ko_loc = sym_loc;
            }
        }
    }

    if sym_ko_loc != NULL_LOC {
        sym_board.set_simple_ko_loc(sym_ko_loc);
    }
    sym_board
}

/// Mark symmetrically-equivalent move locations as duplicates.
///
/// Returns a tuple `(is_sym_dup_loc, valid_symmetries)`. `is_sym_dup_loc` has
/// length [`MAX_ARR_SIZE`] and is `true` for every location that is a duplicate
/// of an earlier canonical location. `valid_symmetries` lists all symmetries
/// (including identity) that leave the current position unchanged, taking into
/// account ko/superko bans and encore state.
pub fn mark_duplicate_move_locs(
    board: &Board,
    hist: &BoardHistory,
    only_symmetries: Option<&[i32]>,
    avoid_moves: &[i32],
) -> (Vec<bool>, Vec<i32>) {
    let mut is_sym_dup_loc = vec![false; MAX_ARR_SIZE];
    let mut valid_symmetries = Vec::with_capacity(NUM_SYMMETRIES as usize);
    valid_symmetries.push(0);

    // If any move is banned by ko or superko, the position is not symmetric.
    if board.ko_loc != NULL_LOC {
        return (is_sym_dup_loc, valid_symmetries);
    }
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            if hist.is_super_ko_banned(loc) {
                return (is_sym_dup_loc, valid_symmetries);
            }
        }
    }

    let symmetry_search_upper_bound = if board.x_size == board.y_size {
        NUM_SYMMETRIES
    } else {
        NUM_SYMMETRIES_WITHOUT_TRANSPOSE
    };

    for symmetry in 1..symmetry_search_upper_bound {
        if let Some(only) = only_symmetries {
            if !only.contains(&symmetry) {
                continue;
            }
        }

        let mut is_board_sym = true;
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size);
                let sym_loc = get_sym_loc(loc, board, symmetry);
                let stone_sym = board.colors[loc as usize] == board.colors[sym_loc as usize];
                let ko_recap_sym = if hist.encore_phase > 0 {
                    hist.is_ko_recap_blocked(loc) == hist.is_ko_recap_blocked(sym_loc)
                } else {
                    true
                };
                let second_encore_sym = if hist.encore_phase == 2 {
                    hist.second_encore_start_color(loc) == hist.second_encore_start_color(sym_loc)
                } else {
                    true
                };
                if !stone_sym || !ko_recap_sym || !second_encore_sym {
                    is_board_sym = false;
                    break;
                }
            }
            if !is_board_sym {
                break;
            }
        }
        if is_board_sym {
            valid_symmetries.push(symmetry);
        }
    }

    let avoid = |loc: Loc| -> bool { avoid_moves.get(loc as usize).copied().unwrap_or(0) > 0 };

    if hist.presumed_next_move_pla == P_BLACK {
        for x in (0..board.x_size).rev() {
            for y in 0..board.y_size {
                let loc = location::get_loc(x, y, board.x_size);
                if avoid(loc) {
                    continue;
                }
                for &symmetry in &valid_symmetries {
                    if symmetry == 0 {
                        continue;
                    }
                    let sym_loc = get_sym_loc(loc, board, symmetry);
                    if !is_sym_dup_loc[loc as usize] && loc != sym_loc {
                        is_sym_dup_loc[sym_loc as usize] = true;
                    }
                }
            }
        }
    } else {
        for x in 0..board.x_size {
            for y in (0..board.y_size).rev() {
                let loc = location::get_loc(x, y, board.x_size);
                if avoid(loc) {
                    continue;
                }
                for &symmetry in &valid_symmetries {
                    if symmetry == 0 {
                        continue;
                    }
                    let sym_loc = get_sym_loc(loc, board, symmetry);
                    if !is_sym_dup_loc[loc as usize] && loc != sym_loc {
                        is_sym_dup_loc[sym_loc as usize] = true;
                    }
                }
            }
        }
    }

    (is_sym_dup_loc, valid_symmetries)
}

/// For each symmetry, return a metric describing how much `board` would differ
/// from `other` if that symmetry were applied to `board`.
///
/// The returned array has length [`NUM_SYMMETRIES`]. For non-square boards,
/// transpositions are not evaluated and left at `max_difference_to_report`.
pub fn get_symmetry_differences(
    board: &Board,
    other: &Board,
    max_difference_to_report: f64,
) -> [f64; NUM_SYMMETRIES as usize] {
    let mut differences = [max_difference_to_report; NUM_SYMMETRIES as usize];

    if board.x_size != other.x_size || board.y_size != other.y_size {
        return differences;
    }

    let num_symmetries = if board.x_size == board.y_size {
        NUM_SYMMETRIES
    } else {
        NUM_SYMMETRIES_WITHOUT_TRANSPOSE
    };

    for symmetry in 0..num_symmetries {
        differences[symmetry as usize] =
            get_symmetry_difference(board, other, symmetry, max_difference_to_report);
    }
    differences
}

fn get_symmetry_difference(
    board: &Board,
    other: &Board,
    symmetry: i32,
    max_difference_to_report: f64,
) -> f64 {
    let mut diff = 0.0;
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            let sym_loc = get_sym_loc(loc, board, symmetry);
            if board.colors[loc as usize] != other.colors[sym_loc as usize] {
                if board.colors[loc as usize] == C_EMPTY
                    || other.colors[sym_loc as usize] == C_EMPTY
                {
                    diff += 1.0;
                } else {
                    diff += 3.0;
                }
                if diff > max_difference_to_report {
                    return max_difference_to_report;
                }
            }
        }
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, P_WHITE, location};
    use crate::rules::Rules;

    #[test]
    fn test_identity() {
        let board = Board::new(9, 9);
        let loc = location::get_loc(4, 4, board.x_size);
        assert_eq!(get_sym_loc(loc, &board, 0), loc);
    }

    #[test]
    fn test_flip_x_roundtrip() {
        let board = Board::new(9, 9);
        let loc = location::get_loc(2, 3, board.x_size);
        let sym = get_sym_loc(loc, &board, 2);
        let back = get_sym_loc(sym, &board, 2);
        assert_eq!(loc, back);
    }

    #[test]
    fn test_transpose_swaps_size() {
        let loc = get_sym_loc_with_size(
            location::get_loc(1, 2, 9),
            9,
            9,
            4, // transpose
        );
        assert_eq!(location::get_x(loc, 9), 2);
        assert_eq!(location::get_y(loc, 9), 1);
    }

    #[test]
    fn test_pass_and_null_untouched() {
        let board = Board::new(9, 9);
        assert_eq!(get_sym_loc(PASS_LOC, &board, 7), PASS_LOC);
        assert_eq!(get_sym_loc(NULL_LOC, &board, 7), NULL_LOC);
    }

    #[test]
    fn test_compose_identity() {
        assert_eq!(compose(0, 3), 3);
    }

    #[test]
    fn test_invert_involution() {
        for sym in 0..8 {
            assert_eq!(invert(invert(sym)), sym);
        }
    }

    #[test]
    fn test_get_sym_board_roundtrip() {
        let mut board = Board::new(5, 5);
        board.set_stone_fail_if_no_libs(location::get_loc(1, 2, 5), P_BLACK);
        board.set_stone_fail_if_no_libs(location::get_loc(3, 4, 5), P_WHITE);

        for sym in 0..8 {
            let sym_board = get_sym_board(&board, sym);
            let back = get_sym_board(&sym_board, invert(sym));
            assert!(
                board.is_equal_for_testing(&back, true, true),
                "roundtrip failed for symmetry {}",
                sym
            );
        }
    }

    #[test]
    fn test_get_sym_board_transposes_dimensions() {
        let mut board = Board::new(3, 5);
        board.set_stone_fail_if_no_libs(location::get_loc(2, 4, 3), P_BLACK);
        let sym_board = get_sym_board(&board, 4); // transpose
        assert_eq!(sym_board.x_size, 5);
        assert_eq!(sym_board.y_size, 3);
    }

    #[test]
    fn test_mark_duplicate_move_locs_empty_board() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let (dups, valid) = mark_duplicate_move_locs(&board, &hist, None, &[]);
        assert_eq!(valid.len(), 8);
        // For black-to-move the canonical traversal starts at high x, so (4,0)
        // is canonical and the other corners are duplicates of it.
        assert!(!dups[location::get_loc(4, 0, 5) as usize]);
        assert!(dups[location::get_loc(0, 0, 5) as usize]);
        assert!(dups[location::get_loc(0, 4, 5) as usize]);
        assert!(dups[location::get_loc(4, 4, 5) as usize]);
    }

    #[test]
    fn test_mark_duplicate_move_locs_ko_breaks_symmetry() {
        let mut board = Board::new(5, 5);
        board.set_simple_ko_loc(location::get_loc(2, 2, 5));
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let (dups, valid) = mark_duplicate_move_locs(&board, &hist, None, &[]);
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0], 0);
        assert!(!dups.iter().any(|&d| d));
    }

    #[test]
    fn test_copy_inputs_with_symmetry_flip_x_nchw() {
        let n = 1;
        let h = 2;
        let w = 3;
        let c = 1;
        let src: Vec<f32> = (0..(n * c * h * w)).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; src.len()];
        // symmetry 2 flips x
        copy_inputs_with_symmetry(&src, &mut dst, n, h, w, c, false, 2);
        // NCHW layout: index = n*c*h*w + c*h*w + h*w + w
        // row 0: 0 1 2 -> flipped -> 2 1 0
        // row 1: 3 4 5 -> flipped -> 5 4 3
        assert_eq!(dst, vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_copy_inputs_with_symmetry_transpose_nhwc() {
        let n = 1;
        let h = 2;
        let w = 2;
        let c = 1;
        let src: Vec<f32> = (0..(n * h * w * c)).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; src.len()];
        // symmetry 4 transposes x/y on square board
        copy_inputs_with_symmetry(&src, &mut dst, n, h, w, c, true, 4);
        // NHWC layout original: [0, 1; 2, 3] -> transposed [0, 2; 1, 3]
        assert_eq!(dst, vec![0.0, 2.0, 1.0, 3.0]);
    }
}
