//! Local pattern hashing around a board point.
//!
//! Corresponds to `cpp/search/localpattern.h` and `cpp/search/localpattern.cpp`.

use kata_core::hash::Hash128;
use kata_core::rng::Rand;
use kata_game::board::{
    Board, C_BLACK, C_WHITE, Loc, NULL_LOC, NUM_BOARD_COLORS, P_BLACK, P_WHITE, PASS_LOC, Player,
    location,
};

fn is_transpose(symmetry: i32) -> bool {
    (symmetry & 0x4) != 0
}

fn is_flip_x(symmetry: i32) -> bool {
    (symmetry & 0x2) != 0
}

fn is_flip_y(symmetry: i32) -> bool {
    (symmetry & 0x1) != 0
}

/// Hashes a small rectangular neighborhood around a board point.
#[derive(Default)]
pub struct LocalPatternHasher {
    x_size: i32,
    y_size: i32,
    zobrist_local_pattern: Vec<Hash128>,
    zobrist_pla: Vec<Hash128>,
    zobrist_atari: Vec<Hash128>,
}

impl LocalPatternHasher {
    /// Create a new uninitialized hasher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the hasher for a board size and random source.
    ///
    /// Panics if `x` or `y` are not positive odd integers.
    pub fn init(&mut self, x: i32, y: i32, rand: &mut Rand) {
        assert!(
            x > 0 && x % 2 == 1,
            "LocalPatternHasher xSize must be positive and odd"
        );
        assert!(
            y > 0 && y % 2 == 1,
            "LocalPatternHasher ySize must be positive and odd"
        );

        self.x_size = x;
        self.y_size = y;
        let area = (x * y) as usize;
        self.zobrist_local_pattern = vec![Hash128::default(); NUM_BOARD_COLORS * area];
        self.zobrist_pla = vec![Hash128::default(); NUM_BOARD_COLORS];
        self.zobrist_atari = vec![Hash128::default(); area];

        for c in 0..NUM_BOARD_COLORS {
            for dy in 0..y {
                for dx in 0..x {
                    let h0 = rand.next_u64();
                    let h1 = rand.next_u64();
                    self.zobrist_local_pattern[c * area + (dy * x + dx) as usize] =
                        Hash128::new(h0, h1);
                }
            }
        }
        for c in 0..NUM_BOARD_COLORS {
            let h0 = rand.next_u64();
            let h1 = rand.next_u64();
            self.zobrist_pla[c] = Hash128::new(h0, h1);
        }
        for dy in 0..y {
            for dx in 0..x {
                let h0 = rand.next_u64();
                let h1 = rand.next_u64();
                self.zobrist_atari[(dy * x + dx) as usize] = Hash128::new(h0, h1);
            }
        }
    }

    /// Hash the local pattern around `loc` for player `pla`.
    pub fn get_hash(&self, board: &Board, loc: Loc, pla: Player) -> Hash128 {
        let mut hash = self.zobrist_pla[pla as usize];

        if loc != PASS_LOC && loc != NULL_LOC {
            let x_radius = self.x_size / 2;
            let y_radius = self.y_size / 2;
            let x_center = self.x_size / 2;
            let y_center = self.y_size / 2;

            let x = location::get_x(loc, board.x_size);
            let y = location::get_y(loc, board.y_size);

            let mut dx_min = -x_radius;
            let mut dx_max = x_radius;
            let mut dy_min = -y_radius;
            let mut dy_max = y_radius;

            if x < x_radius {
                dx_min = -x;
            } else if x >= board.x_size - x_radius {
                dx_max = board.x_size - 1 - x;
            }
            if y < y_radius {
                dy_min = -y;
            } else if y >= board.y_size - y_radius {
                dy_max = board.y_size - 1 - y;
            }

            let area = (self.x_size * self.y_size) as usize;
            for dy in dy_min..=dy_max {
                for dx in dx_min..=dx_max {
                    let loc2 = location::get_loc(x + dx, y + dy, board.x_size);
                    let y2 = dy + y_center;
                    let x2 = dx + x_center;
                    let xy2 = (y2 * self.x_size + x2) as usize;
                    let color = board.colors[loc2 as usize] as usize;
                    hash ^= self.zobrist_local_pattern[color * area + xy2];
                    if (color == C_BLACK as usize || color == C_WHITE as usize)
                        && board.get_num_liberties(loc2) == 1
                    {
                        hash ^= self.zobrist_atari[xy2];
                    }
                }
            }
        }

        hash
    }

    /// Hash the local pattern after applying a symmetry to the neighborhood.
    ///
    /// `symmetry` uses the same encoding as KataGo: flipY (bit 0), flipX (bit 1), transpose (bit 2).
    pub fn get_hash_with_sym(
        &self,
        board: &Board,
        loc: Loc,
        pla: Player,
        symmetry: i32,
        flip_colors: bool,
    ) -> Hash128 {
        let sym_pla = if flip_colors {
            match pla {
                P_BLACK => P_WHITE,
                P_WHITE => P_BLACK,
                _ => pla,
            }
        } else {
            pla
        };
        let mut hash = self.zobrist_pla[sym_pla as usize];

        if loc != PASS_LOC && loc != NULL_LOC {
            let transpose = is_transpose(symmetry);
            let flip_x = is_flip_x(symmetry);
            let flip_y = is_flip_y(symmetry);

            let x_radius = self.x_size / 2;
            let y_radius = self.y_size / 2;
            let x_center = self.x_size / 2;
            let y_center = self.y_size / 2;

            let x = location::get_x(loc, board.x_size);
            let y = location::get_y(loc, board.y_size);

            let mut dx_min = -x_radius;
            let mut dx_max = x_radius;
            let mut dy_min = -y_radius;
            let mut dy_max = y_radius;

            if x < x_radius {
                dx_min = -x;
            } else if x >= board.x_size - x_radius {
                dx_max = board.x_size - 1 - x;
            }
            if y < y_radius {
                dy_min = -y;
            } else if y >= board.y_size - y_radius {
                dy_max = board.y_size - 1 - y;
            }

            let area = (self.x_size * self.y_size) as usize;
            for dy in dy_min..=dy_max {
                for dx in dx_min..=dx_max {
                    let loc2 = location::get_loc(x + dx, y + dy, board.x_size);
                    let y2 = dy + y_center;
                    let x2 = dx + x_center;

                    let mut sym_x2 = if flip_x { self.x_size - x2 - 1 } else { x2 };
                    let mut sym_y2 = if flip_y { self.y_size - y2 - 1 } else { y2 };
                    let sym_xy2 = if transpose {
                        std::mem::swap(&mut sym_x2, &mut sym_y2);
                        (sym_y2 * self.y_size + sym_x2) as usize
                    } else {
                        (sym_y2 * self.x_size + sym_x2) as usize
                    };

                    let color = board.colors[loc2 as usize];
                    let sym_color = if color == C_BLACK || color == C_WHITE {
                        if flip_colors {
                            match color {
                                C_BLACK => C_WHITE,
                                C_WHITE => C_BLACK,
                                c => c,
                            }
                        } else {
                            color
                        }
                    } else {
                        color
                    } as usize;

                    hash ^= self.zobrist_local_pattern[sym_color * area + sym_xy2];
                    if (color == C_BLACK || color == C_WHITE) && board.get_num_liberties(loc2) == 1
                    {
                        hash ^= self.zobrist_atari[sym_xy2];
                    }
                }
            }
        }

        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{C_EMPTY, get_opp};

    fn seeded_rand() -> Rand {
        let mut rand = Rand::new();
        rand.init_from_u64(12345);
        rand
    }

    fn empty_board() -> Board {
        Board::new(9, 9)
    }

    #[test]
    fn test_init_requires_odd_sizes() {
        let mut hasher = LocalPatternHasher::new();
        hasher.init(5, 5, &mut seeded_rand());
        assert_eq!(hasher.x_size, 5);
        assert_eq!(hasher.y_size, 5);
    }

    #[test]
    #[should_panic]
    fn test_init_rejects_even_size() {
        let mut hasher = LocalPatternHasher::new();
        hasher.init(4, 5, &mut seeded_rand());
    }

    #[test]
    fn test_hash_stable() {
        let board = empty_board();
        let mut hasher = LocalPatternHasher::new();
        hasher.init(3, 3, &mut seeded_rand());
        let loc = location::get_loc(3, 3, board.x_size);
        let h1 = hasher.get_hash(&board, loc, P_BLACK);
        let h2 = hasher.get_hash(&board, loc, P_BLACK);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_pass_and_null_loc_only_depend_on_player() {
        let board = empty_board();
        let mut hasher = LocalPatternHasher::new();
        hasher.init(3, 3, &mut seeded_rand());
        assert_eq!(
            hasher.get_hash(&board, PASS_LOC, P_BLACK),
            hasher.get_hash(&board, NULL_LOC, P_BLACK)
        );
        assert_ne!(
            hasher.get_hash(&board, PASS_LOC, P_BLACK),
            hasher.get_hash(&board, PASS_LOC, P_WHITE)
        );
    }

    #[test]
    fn test_hash_changes_with_stone() {
        let mut board = empty_board();
        let mut hasher = LocalPatternHasher::new();
        hasher.init(3, 3, &mut seeded_rand());
        let loc = location::get_loc(3, 3, board.x_size);

        let empty_hash = hasher.get_hash(&board, loc, P_BLACK);
        board.set_stone(loc, C_BLACK);
        let stone_hash = hasher.get_hash(&board, loc, P_BLACK);
        assert_ne!(empty_hash, stone_hash);
    }

    #[test]
    fn test_atari_detected() {
        // Place a black stone surrounded on three sides by white; it has exactly one liberty.
        let mut board = empty_board();
        let center = location::get_loc(2, 2, board.x_size);
        board.set_stone(center, C_BLACK);
        board.set_stone(location::get_loc(1, 2, board.x_size), C_WHITE);
        board.set_stone(location::get_loc(3, 2, board.x_size), C_WHITE);
        board.set_stone(location::get_loc(2, 1, board.x_size), C_WHITE);

        assert_eq!(board.get_num_liberties(center), 1);

        // The hash should still be computed without panicking and should differ from an empty board.
        let mut hasher = LocalPatternHasher::new();
        hasher.init(3, 3, &mut seeded_rand());
        assert_ne!(
            hasher.get_hash(&board, center, P_BLACK),
            hasher.get_hash(&empty_board(), center, P_BLACK)
        );
    }

    #[test]
    fn test_identity_symmetry_matches_original() {
        let mut board = empty_board();
        let mut hasher = LocalPatternHasher::new();
        hasher.init(3, 3, &mut seeded_rand());
        let loc = location::get_loc(4, 4, board.x_size);
        board.set_stone(loc, C_BLACK);
        board.set_stone(location::get_loc(3, 4, board.x_size), C_WHITE);

        let h = hasher.get_hash(&board, loc, P_BLACK);
        let h_sym = hasher.get_hash_with_sym(&board, loc, P_BLACK, 0, false);
        assert_eq!(h, h_sym);
    }

    #[test]
    fn test_one_by_one_symmetry_variants_match() {
        // With a 1x1 window every symmetry maps the only cell to itself.
        let mut board = empty_board();
        let mut hasher = LocalPatternHasher::new();
        hasher.init(1, 1, &mut seeded_rand());
        let loc = location::get_loc(4, 4, board.x_size);
        board.set_stone(loc, C_BLACK);

        let h = hasher.get_hash(&board, loc, P_BLACK);
        for symmetry in 0..8 {
            assert_eq!(
                h,
                hasher.get_hash_with_sym(&board, loc, P_BLACK, symmetry, false)
            );
        }
    }

    #[test]
    fn test_flip_colors_changes_local_color() {
        // A single black stone hashed as black with color-flip should equal a single
        // white stone hashed as white, since both the player and the on-board color swap.
        let mut black_board = empty_board();
        let loc = location::get_loc(4, 4, black_board.x_size);
        black_board.set_stone(loc, C_BLACK);

        let mut white_board = empty_board();
        white_board.set_stone(loc, C_WHITE);

        let mut hasher = LocalPatternHasher::new();
        hasher.init(1, 1, &mut seeded_rand());

        let h_black_flipped = hasher.get_hash_with_sym(&black_board, loc, P_BLACK, 0, true);
        let h_white = hasher.get_hash(&white_board, loc, P_WHITE);
        assert_eq!(h_black_flipped, h_white);
    }

    #[test]
    fn test_color_flip_uses_opponent_color() {
        assert_eq!(get_opp(C_BLACK), C_WHITE);
        assert_eq!(get_opp(C_WHITE), C_BLACK);
        assert_eq!(get_opp(C_EMPTY), 3); // C_WALL
    }
}
