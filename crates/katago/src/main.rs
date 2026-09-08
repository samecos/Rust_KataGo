//! KataGo Rust CLI entry point.

mod args;
mod cli;
mod cmd;

use std::process::ExitCode;

fn main() -> ExitCode {
    args::make_cout_and_cerr_accept_utf8();
    let args = args::get_command_line_args_utf8();

    // Run the command on a worker thread with a large stack. The default
    // main-thread stack on Windows is small (1 MiB), and engine startup
    // (Zobrist table init) plus deep search recursion overflow it.
    let handle = std::thread::Builder::new()
        .name("katago-main".to_string())
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run(args))
        .expect("failed to spawn main worker thread");
    let code = handle.join().expect("main worker thread panicked");

    ExitCode::from(code as u8)
}

fn run(args: Vec<String>) -> i32 {
    let program = args.first().map(|s| s.as_str()).unwrap_or("katago-rs");
    let (subcommand, rest) = if args.len() >= 2 {
        (args[1].as_str(), &args[2..])
    } else {
        print_usage(program);
        return 1;
    };

    match subcommand {
        "gtp" => cmd::gtp::gtp(rest),
        "analysis" => cmd::analysis::analysis(rest),
        "benchmark" => cmd::benchmark::benchmark(rest),
        "genconfig" => cmd::genconfig::genconfig(rest),
        "nnbench" => cmd::nnbench::nnbench(rest),
        "nnworker" => cmd::nnworker::nnworker(rest),
        "cuda-fingerprint" => cmd::cuda_fingerprint::cuda_fingerprint(rest),
        "contribute" => cmd::contribute::contribute(rest),
        "evalsgf" => cmd::eval_sgf::evalsgf(rest),
        "gatekeeper" => cmd::gatekeeper::gatekeeper(rest),
        "genbook" => cmd::gen_book::genbook(rest),
        "writebook" => cmd::gen_book::writebook(rest),
        "checkbook" => cmd::gen_book::checkbook(rest),
        "booktoposes" => cmd::gen_book::booktoposes(rest),
        "comparebooks" => cmd::gen_book::comparebooks(rest),
        "findbookbottlenecks" => cmd::gen_book::findbookbottlenecks(rest),
        "testgpuerror" => cmd::gpu_test::testgpuerror(rest),
        "sandbox" => cmd::sandbox::sandbox(rest),
        "match" => cmd::r#match::match_cmd(rest),
        "tune" => cmd::tune::tune(rest),
        "writetrainingdata" => cmd::write_training_data::writetrainingdata(rest),
        "selfplay" => cmd::selfplay::selfplay(rest),
        "samplesgfs" => cmd::start_poses::samplesgfs(rest),
        "dataminesgfs" => cmd::start_poses::dataminesgfs(rest),
        "trystartposes" => cmd::start_poses::trystartposes(rest),
        "viewstartposes" => cmd::start_poses::viewstartposes(rest),
        "checksgfhintpolicy" => cmd::start_poses::checksgfhintpolicy(rest),
        "genposesfromselfplayinit" => cmd::start_poses::genposesfromselfplayinit(rest),
        "runtests" => cmd::test::runtests(rest),
        "misc" => dispatch_misc(rest),
        "help" | "--help" | "-h" => {
            print_usage(program);
            0
        }
        _ => {
            eprintln!("Unknown subcommand: {}", subcommand);
            print_usage(program);
            1
        }
    }
}

fn dispatch_misc(args: &[String]) -> i32 {
    let (sub, rest) = args
        .split_first()
        .map(|(s, r)| (s.as_str(), r))
        .unwrap_or(("", &[]));
    match sub {
        "printclockinfo" => cmd::misc::print_clock_info(rest),
        "sampleinitializations" => cmd::misc::sample_initializations(rest),
        "evalrandominits" => cmd::misc::eval_random_inits(rest),
        "searchentropyanalysis" => cmd::misc::search_entropy_analysis(rest),
        _ => {
            eprintln!("Unknown misc subcommand: {}", sub);
            eprintln!(
                "Known misc subcommands: printclockinfo, sampleinitializations, evalrandominits, searchentropyanalysis"
            );
            1
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} <subcommand> [options]", program);
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  gtp                         Run the GTP engine (placeholder)");
    eprintln!("  analysis                    Run the JSON analysis engine");
    eprintln!("  benchmark                   Run NN/search benchmark");
    eprintln!("  genconfig                   Interactively generate and tune a GTP config");
    eprintln!("  nnbench                     Pure NN forward throughput benchmark (cuda)");
    eprintln!("  nnworker                    Connect to Go Server as a pure NN gRPC worker");
    eprintln!(
        "  contribute                  Contribute games to distributed training (not enabled)"
    );
    eprintln!("  evalsgf                     Run a search on a position from an SGF file");
    eprintln!(
        "  gatekeeper                  Test candidate neural nets against an accepted baseline"
    );
    eprintln!("  genbook                     Generate/opening book commands (partial)");
    eprintln!("    writebook / checkbook     Load a book and export HTML or verify integrity");
    eprintln!("    booktoposes / comparebooks / findbookbottlenecks (not implemented)");
    eprintln!("  match                       Run a match or tournament between nets");
    eprintln!("  testgpuerror                Test neural net GPU/backend error vs FP32");
    eprintln!("  sandbox                     Backend-specific smoke test (TensorRT-only in C++)");
    eprintln!("  selfplay                    Generate training data via self play");
    eprintln!("  samplesgfs / dataminesgfs / trystartposes / viewstartposes");
    eprintln!("  checksgfhintpolicy / genposesfromselfplayinit");
    eprintln!("                              Starting-position and SGF sampling (not implemented)");
    eprintln!("  runtests                    Run internal integration tests (placeholder)");
    eprintln!("  tune                        Perform GPU tuning for OpenCL (placeholder)");
    eprintln!(
        "  writetrainingdata           Convert SGFs/sources into training data (not implemented)"
    );
    eprintln!("  misc <sub>                  Miscellaneous commands");
    eprintln!("    printclockinfo");
    eprintln!("    sampleinitializations");
    eprintln!("    evalrandominits");
    eprintln!("    searchentropyanalysis");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_misc_unknown_returns_one() {
        assert_eq!(dispatch_misc(&["unknown".to_string()]), 1);
    }

    #[test]
    fn test_dispatch_misc_print_clock_info() {
        assert_eq!(dispatch_misc(&["printclockinfo".to_string()]), 0);
    }
}
