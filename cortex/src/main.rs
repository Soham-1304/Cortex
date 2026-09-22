fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if let Err(e) = cortex::cli::run(argv) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
