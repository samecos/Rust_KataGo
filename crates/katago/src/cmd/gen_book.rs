//! Opening book commands.
//!
//! Corresponds to the book-related subcommands in `cpp/command/genbook.cpp`:
//! `genbook`, `writebook`, `checkbook`, `booktoposes`, `comparebooks`, and
//! `findbookbottlenecks`.

use clap::Parser;

use kata_book::{Book, BookHash, BookParams};
use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK};
use kata_game::rules::Rules;

fn make_logger() -> &'static Logger {
    Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: true,
            log_to_stderr: true,
            log_time: true,
        },
        None,
    )))
}

fn io_error(s: String) -> StringError {
    StringError::new(s)
}

/// CLI arguments for the `checkbook` subcommand.
#[derive(Parser, Debug, Clone)]
struct CheckBookArgs {
    /// Book file to verify.
    #[arg(long = "book-file", value_name = "FILE")]
    book_file: String,
}

/// Public CLI entry point for `checkbook`.
pub fn checkbook(args: &[String]) -> i32 {
    match checkbook_impl(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn checkbook_impl(args: &[String]) -> Result<(), StringError> {
    let parsed =
        CheckBookArgs::try_parse_from(std::iter::once(&"checkbook".to_string()).chain(args.iter()))
            .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let logger = make_logger();
    let book = Book::load_from_file(&parsed.book_file)
        .map_err(|e| io_error(format!("Could not load book file: {}", e.0)))?;

    logger.write(&format!(
        "Loaded preexisting book with {} nodes from {}",
        book.size(),
        parsed.book_file
    ));
    logger.write(&format!("Book version = {}", book.book_version));

    let mut failures = 0usize;
    let mut warnings = 0usize;

    for (i, node) in book.get_all_nodes().iter().enumerate() {
        let (hist, _) = match node.get_board_history_reaching_here(&book) {
            Some(h) => h,
            None => {
                warnings += 1;
                eprintln!(
                    "Warning: could not reconstruct history reaching node with hash {}",
                    node.hash().to_string()
                );
                continue;
            }
        };

        let recomputed = BookHash::get_hash_and_symmetry(&hist, book.rep_bound, book.book_version);
        if recomputed.hash != node.hash() {
            failures += 1;
            eprintln!(
                "Integrity check failure: node hash {} recomputed as {}",
                node.hash().to_string(),
                recomputed.hash.to_string()
            );
        }

        if (i + 1) % 10_000 == 0 {
            logger.write(&format!("Checked {} nodes", i + 1));
        }
    }

    logger.write(&format!(
        "Finished checking {} nodes ({} failures, {} warnings)",
        book.size(),
        failures,
        warnings
    ));

    if failures > 0 {
        return Err(io_error(format!(
            "Book integrity check failed with {} mismatches",
            failures
        )));
    }
    Ok(())
}

/// CLI arguments for the `writebook` subcommand.
#[derive(Parser, Debug, Clone)]
struct WriteBookArgs {
    #[command(flatten)]
    common: crate::cli::CommonArgs,

    /// Book file to load.
    #[arg(long = "book-file", value_name = "FILE")]
    book_file: String,

    /// Directory to export HTML to.
    #[arg(long = "html-dir", value_name = "DIR")]
    html_dir: String,

    /// Label for the rules link.
    #[arg(long = "rules-label", value_name = "LABEL", default_value = "")]
    rules_label: String,

    /// URL for the rules link.
    #[arg(long = "rules-link", value_name = "URL", default_value = "")]
    rules_link: String,

    /// Enable HTML development mode.
    #[arg(long = "html-dev-mode")]
    html_dev_mode: bool,

    /// Minimum visits for a node to be exported to HTML.
    #[arg(long = "html-min-visits", value_name = "N", default_value_t = 0.0)]
    html_min_visits: f64,

    /// Maximum visits for book expansion.
    #[arg(long = "max-visits", value_name = "N", default_value_t = 1000)]
    max_visits: i64,

    /// Maximum visits for leaf nodes.
    #[arg(
        long = "max-visits-for-leaves",
        value_name = "N",
        default_value_t = 500
    )]
    max_visits_for_leaves: i64,
}

/// Public CLI entry point for `writebook`.
pub fn writebook(args: &[String]) -> i32 {
    match writebook_impl(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn writebook_impl(args: &[String]) -> Result<(), StringError> {
    let parsed =
        WriteBookArgs::try_parse_from(std::iter::once(&"writebook".to_string()).chain(args.iter()))
            .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let logger = make_logger();
    let mut book = Book::load_from_file(&parsed.book_file)
        .map_err(|e| io_error(format!("Could not load book file: {}", e.0)))?;

    if !parsed.common.config.is_empty() {
        let mut cfg = parsed.common.get_config("gtp_example.cfg")?;
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        let params =
            BookParams::load_from_cfg(&cfg, parsed.max_visits, parsed.max_visits_for_leaves)
                .map_err(|e| io_error(format!("Could not load book params: {}", e)))?;
        book.set_params(params);
    }

    book.recompute_everything();

    logger.write(&format!("EXPORTING HTML TO {}", parsed.html_dir));

    let num_files = book
        .export_to_html_dir(
            &parsed.html_dir,
            &parsed.rules_label,
            &parsed.rules_link,
            parsed.html_dev_mode,
            parsed.html_min_visits,
        )
        .map_err(|e| io_error(format!("Could not export HTML: {}", e.0)))?;

    logger.write(&format!("Done exporting, exported {} files", num_files));
    Ok(())
}

fn report_not_implemented(name: &str) -> Result<(), StringError> {
    // Keep the book crate linked: create an empty book on each call.
    let _book = Book::new(
        2,
        Board::new(7, 7),
        Rules::default(),
        P_BLACK,
        3,
        BookParams::default(),
    );
    println!(
        "{}: Opening book support is not yet implemented in the Rust port.",
        name
    );
    Ok(())
}

/// Public CLI entry point for `genbook`. Not yet implemented.
pub fn genbook(_args: &[String]) -> i32 {
    match report_not_implemented("genbook") {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Public CLI entry point for `booktoposes`. Not yet implemented.
pub fn booktoposes(_args: &[String]) -> i32 {
    match report_not_implemented("booktoposes") {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Public CLI entry point for `comparebooks`. Not yet implemented.
pub fn comparebooks(_args: &[String]) -> i32 {
    match report_not_implemented("comparebooks") {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Public CLI entry point for `findbookbottlenecks`. Not yet implemented.
pub fn findbookbottlenecks(_args: &[String]) -> i32 {
    match report_not_implemented("findbookbottlenecks") {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_book_file() -> (tempfile::NamedTempFile, Book) {
        let file = tempfile::NamedTempFile::new().unwrap();
        let book = Book::new(
            2,
            Board::new(7, 7),
            Rules::default(),
            P_BLACK,
            3,
            BookParams::default(),
        );
        book.save_to_file(file.path()).unwrap();
        (file, book)
    }

    #[test]
    fn test_genbook_reports_not_implemented() {
        assert_eq!(genbook(&[]), 0);
    }

    #[test]
    fn test_writebook_exports_html() {
        let (book_file, _book) = make_test_book_file();
        let html_dir = tempfile::tempdir().unwrap();
        let args = vec![
            "--book-file".to_string(),
            book_file.path().to_str().unwrap().to_string(),
            "--html-dir".to_string(),
            html_dir.path().to_str().unwrap().to_string(),
        ];
        assert_eq!(writebook(&args), 0);
        assert!(html_dir.path().join("book.js").exists());
        assert!(html_dir.path().join("book.css").exists());
    }

    #[test]
    fn test_checkbook_succeeds_on_fresh_book() {
        let (book_file, _book) = make_test_book_file();
        let args = vec![
            "--book-file".to_string(),
            book_file.path().to_str().unwrap().to_string(),
        ];
        assert_eq!(checkbook(&args), 0);
    }

    #[test]
    fn test_booktoposes_reports_not_implemented() {
        assert_eq!(booktoposes(&[]), 0);
    }

    #[test]
    fn test_comparebooks_reports_not_implemented() {
        assert_eq!(comparebooks(&[]), 0);
    }

    #[test]
    fn test_findbookbottlenecks_reports_not_implemented() {
        assert_eq!(findbookbottlenecks(&[]), 0);
    }
}
