//! Export calibration positions through the engine's SGF parser. Comments are
//! never interpreted as moves. Branch selection matches GTP loadsgf (deepest).
use kata_data::sgf::CompactSgf;
use kata_game::{
    board::{Board, P_BLACK, P_WHITE, PASS_LOC, location},
    history::BoardHistory,
    rules::Rules,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: sgf_worker_fixture INPUT.sgf OUTPUT.json".into());
    }
    let bytes = std::fs::read(&args[1])?;
    let sgf = CompactSgf::parse_file(&args[1]).map_err(|e| e.0)?;
    if (sgf.x_size, sgf.y_size) != (19, 19) || !sgf.placements.is_empty() {
        return Err("fixture exporter requires 19x19 with no setup stones".into());
    }
    let chinese = Rules::parse_rules("chinese").map_err(|e| e.0)?;
    let rules = sgf
        .get_rules_or_fail_allow_unspecified(&chinese)
        .map_err(|e| e.0)?;
    if rules != chinese || rules.komi_f32() != 7.5 {
        return Err("fixture requires Chinese rules and komi 7.5".into());
    }
    let mut board = Board::new(19, 19);
    let mut history = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut moves = Vec::new();
    let mut positions = Vec::new();
    for (i, mv) in sgf.moves.iter().enumerate() {
        let player = if i % 2 == 0 { P_BLACK } else { P_WHITE };
        if mv.pla != player || !history.is_legal(&board, mv.loc, mv.pla) {
            return Err(format!("nonalternating or illegal move at ply {}", i + 1).into());
        }
        history.make_board_move_assume_legal(&mut board, mv.loc, mv.pla);
        let vertex = if mv.loc == PASS_LOC {
            -1
        } else {
            location::get_y(mv.loc, 19) * 19 + location::get_x(mv.loc, 19)
        };
        moves.push((mv.pla, vertex));
        // Omit trivial opening prefixes shared by unrelated games. Each
        // selected history is subsequently evaluated under all 8 symmetries.
        // Include adjacent plies so calibration covers both players to move.
        if i + 1 >= 8 && matches!((i + 1) % 8, 0 | 1) && !history.is_game_finished {
            positions.push(json!({"name":format!("calibration-ply-{:03}",i+1),"moves":moves}));
        }
    }
    if positions.is_empty() {
        return Err("no calibration positions".into());
    }
    let fixture = json!({"source_sgf_sha256":kata_core::hash::sha2::sha256_hex(&bytes),
        "source": "user supplied SGF; native deepest branch; comments omitted", "total_moves": moves.len(),
        "description":"Every eighth legal nonterminal position and its following ply, starting at 8; both players; calibration only", "positions":positions});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer_pretty(&mut file, &fixture)?;
    println!(
        "Exported {} histories from {} moves",
        positions.len(),
        moves.len()
    );
    Ok(())
}
