use clap::{Parser, ValueEnum};
use flate2::read::MultiGzDecoder;
#[cfg(debug_assertions)]
use indicatif::ProgressBar;
#[cfg(debug_assertions)]
use indicatif::ProgressStyle;
use lib_tracepoints::tracepoints_to_fastga_cigar;
#[cfg(debug_assertions)]
use lib_tracepoints::{align_sequences_wfa, cigar_ops_to_cigar_string};
use lib_tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_variable_tracepoints,
    mixed_tracepoints_to_cigar, tracepoints_to_cigar, variable_tracepoints_to_cigar,
    MixedRepresentation,
};
use lib_tracepoints::{CigarPosition, TPBundle};
#[cfg(debug_assertions)]
use lib_wfa2::affine_wavefront::AffineWavefronts;
use log::{error, info};
use rayon::prelude::*;
use rust_htslib::faidx::Reader as FastaReader;
use std::fs::File;
use std::fmt;
use std::io::{self, BufRead, BufReader};

/// Tracepoint representation type
#[derive(Debug, Clone, ValueEnum)]
enum TracepointType {
    /// Standard tracepoints
    Standard,
    /// Mixed representation (preserves S/H/P/N CIGAR operations)
    Mixed,
    /// Variable tracepoints representation
    Variable,
}

impl fmt::Display for TracepointType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TracepointType::Standard => write!(f, "standard"),
            TracepointType::Mixed => write!(f, "mixed"),
            TracepointType::Variable => write!(f, "variable"),
        }
    }
}

/// Common options shared between all commands
#[derive(Parser, Debug)]
struct CommonOpts {
    /// PAF file for alignments (use "-" to read from standard input)
    #[arg(short = 'p', long = "paf")]
    paf: String,

    /// Number of threads to use (default: 4)
    #[arg(long = "threads", default_value_t = 4)]
    threads: usize,

    /// Verbosity level (0 = error, 1 = info, 2 = debug)
    #[arg(short, long, default_value = "0")]
    verbose: u8,
}

/// Structure to hold PAF record information
#[cfg(debug_assertions)]
#[derive(Debug, Clone)]
struct PafRecord {
    query_name: String,
    query_start: usize,
    query_end: usize,
    strand: String,
    target_name: String,
    target_start: usize,
    target_end: usize,
    cigar: String,
    original_line: String,
}

#[derive(Parser, Debug)]
#[command(author, version, about, disable_help_subcommand = true)]
enum Args {
    /// Compression of alignments
    Compress {
        #[clap(flatten)]
        common: CommonOpts,

        /// Use mixed representation (preserves S/H/P/N CIGAR operations)
        #[arg(short = 'm', long = "mixed", default_value_t = false)]
        mixed: bool,

        /// Use variable tracepoints representation
        #[arg(long = "variable", default_value_t = false)]
        variable: bool,

        /// Max-diff value for tracepoints
        #[arg(long, default_value = "32")]
        max_diff: usize,
    },
    /// Decompression of alignments
    Decompress {
        #[clap(flatten)]
        common: CommonOpts,

        /// Tracepoint type: standard, mixed, or variable
        #[arg(long = "type", default_value_t = TracepointType::Standard)]
        tp_type: TracepointType,

        /// FASTA file for query sequences
        #[arg(short = 'q', long = "query-fasta")]
        query_fasta: String,

        /// FASTA file for target sequences
        #[arg(short = 't', long = "target-fasta")]
        target_fasta: String,

        /// Gap penalties in the format mismatch,gap_open1,gap_ext1,gap_open2,gap_ext2
        #[arg(long, default_value = "5,8,2,24,1")]
        penalties: String,
    },
    /// Run debugging mode (only available in debug builds)
    #[cfg(debug_assertions)]
    Debug {
        /// PAF file for alignments (use "-" to read from standard input)
        // #[arg(short = 'p', long = "paf")]
        // paf: Option<String>,

        /// First PAF file for comparison
        #[arg(short = 'p', long = "paf1")]
        paf1: Option<String>,
        
        /// Second PAF file for comparison (optional)
        #[arg(long = "paf2")]
        paf2: Option<String>,

        /// Number of threads to use (default: 4)
        #[arg(long = "threads", default_value_t = 4)]
        threads: usize,

        /// FASTA file for query sequences
        #[arg(short = 'q', long = "query-fasta")]
        query_fasta: Option<String>,

        /// FASTA file for target sequences
        #[arg(short = 't', long = "target-fasta")]
        target_fasta: Option<String>,

        /// Gap penalties in the format mismatch,gap_open1,gap_ext1,gap_open2,gap_ext2
        #[arg(long, default_value = "5,8,2,24,1")]
        penalties: String,

        /// Max-diff value for tracepoints
        #[arg(long, default_value = "32")]
        max_diff: usize,

        /// Verbosity level (0 = error, 1 = info, 2 = debug)
        #[arg(short, long, default_value = "0")]
        verbose: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments.
    let args = Args::parse();

    match args {
        Args::Compress {
            common,
            mixed,
            variable,
            max_diff,
        } => {
            setup_logger(common.verbose);
            let tracepoint_type = if mixed {
                "mixed "
            } else if variable {
                "variable "
            } else {
                ""
            };
            info!("Converting CIGAR to {}tracepoints", tracepoint_type);

            // Set the thread pool size
            rayon::ThreadPoolBuilder::new()
                .num_threads(common.threads)
                .build_global()?;

            // Open the PAF file (or use stdin if "-" is provided).
            let paf_reader = get_paf_reader(&common.paf)?;

            // Process in chunks
            let chunk_size = std::cmp::max(common.threads * 100, 1000);
            let mut lines = Vec::with_capacity(chunk_size);
            for line_result in paf_reader.lines() {
                match line_result {
                    Ok(line) => {
                        if line.trim().is_empty() || line.starts_with('#') {
                            continue;
                        }

                        lines.push(line);

                        if lines.len() >= chunk_size {
                            // Process current chunk in parallel
                            process_compress_chunk(&lines, mixed, variable, max_diff);
                            lines.clear();
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }

            // Process remaining lines
            if !lines.is_empty() {
                process_compress_chunk(&lines, mixed, variable, max_diff);
            }
        }
        Args::Decompress {
            common,
            tp_type,
            query_fasta,
            target_fasta,
            penalties,
        } => {
            setup_logger(common.verbose);
            info!("Converting tracepoints to CIGAR");

            // Set the thread pool size
            rayon::ThreadPoolBuilder::new()
                .num_threads(common.threads)
                .build_global()?;

            // Parse penalties
            let (mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2) = parse_penalties(&penalties)?;

            // Open the PAF file (or use stdin if "-" is provided).
            let paf_reader = get_paf_reader(&common.paf)?;

            // Process in chunks
            let chunk_size = std::cmp::max(common.threads * 100, 1000);
            let mut lines = Vec::with_capacity(chunk_size);
            for line_result in paf_reader.lines() {
                match line_result {
                    Ok(line) => {
                        if line.trim().is_empty() || line.starts_with('#') {
                            continue;
                        }

                        lines.push(line);

                        if lines.len() >= chunk_size {
                            // Process current chunk in parallel
                            process_decompress_chunk(
                                &lines,
                                &tp_type,
                                &query_fasta,
                                &target_fasta,
                                mismatch,
                                gap_open1,
                                gap_ext1,
                                gap_open2,
                                gap_ext2,
                            );
                            lines.clear();
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }

            // Process remaining lines
            if !lines.is_empty() {
                process_decompress_chunk(
                    &lines,
                    &tp_type,
                    &query_fasta,
                    &target_fasta,
                    mismatch,
                    gap_open1,
                    gap_ext1,
                    gap_open2,
                    gap_ext2,
                );
            }
        }
        #[cfg(debug_assertions)]
        Args::Debug {
            paf1,
            paf2,
            threads,
            query_fasta,
            target_fasta,
            penalties,
            max_diff,
            verbose,
        } => {
            setup_logger(verbose);
            
            let (mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2) = parse_penalties(&penalties)?;
            
            if let (Some(paf1), Some(paf2), Some(query_fasta), Some(target_fasta)) =
                (paf1, paf2, query_fasta, target_fasta)
            {
                info!("Comparing two PAF files: {} vs {}", paf1, paf2);
                process_two_paf_files(
                    &paf1, &paf2, &query_fasta, &target_fasta,
                    mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2,
                    max_diff, threads
                )?;
            } else if let (Some(paf1), Some(query_fasta), Some(target_fasta)) =
                (paf1, query_fasta, target_fasta)
            {
                // Single PAF file mode (existing functionality)
                info!("Single PAF file debug mode");
                // ... existing single PAF processing code ...
            } else {
                info!("Running default example");
                // ... existing default example code ...
            }
        }
    }

    Ok(())
}

/// Process a chunk of lines in parallel for debugging
#[cfg(debug_assertions)]
fn process_debug_chunk(
    lines: &[String],
    query_fasta_path: &str,
    target_fasta_path: &str,
    mismatch: i32,
    gap_open1: i32,
    gap_ext1: i32,
    gap_open2: i32,
    gap_ext2: i32,
    max_diff: usize,
) {
    lines.par_iter().for_each(|line| {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping malformed PAF line", line)
            );
            std::process::exit(1);
        }

        let Some(cg_field) = fields.iter().find(|&&s| s.starts_with("cg:Z:")) else {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping CIGAR-less PAF line", line)
            );
            std::process::exit(1);
        };
        let paf_cigar = &cg_field[5..];

        // Parse mandatory PAF fields.
        let query_name = fields[0];
        let query_start: usize = fields[2].parse().unwrap_or_else(|_| {
            error!("Invalid query_start in PAF line");
            std::process::exit(1);
        });
        let query_end: usize = fields[3].parse().unwrap_or_else(|_| {
            error!("Invalid query_end in PAF line");
            std::process::exit(1);
        });
        let strand = fields[4];
        let target_name = fields[5];
        let target_start: usize = fields[7].parse().unwrap_or_else(|_| {
            error!("Invalid target_start in PAF line");
            std::process::exit(1);
        });
        let target_end: usize = fields[8].parse().unwrap_or_else(|_| {
            error!("Invalid target_end in PAF line");
            std::process::exit(1);
        });

        // Create thread-local FASTA readers
        let query_fasta_reader = FastaReader::from_path(query_fasta_path).unwrap_or_else(|e| {
            error!("Error reading query FASTA file: {}", e);
            std::process::exit(1);
        });
        let target_fasta_reader = FastaReader::from_path(target_fasta_path).unwrap_or_else(|e| {
            error!("Error reading target FASTA file: {}", e);
            std::process::exit(1);
        });

        // Fetch query sequence from query FASTA
        let query_seq = if strand == "+" {
            match query_fasta_reader.fetch_seq(query_name, query_start, query_end - 1) {
                Ok(seq) => {
                    let mut seq_vec = seq.to_vec();
                    unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                    seq_vec
                        .iter_mut()
                        .for_each(|byte| *byte = byte.to_ascii_uppercase());
                    seq_vec
                }
                Err(e) => {
                    error!("Failed to fetch query sequence: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            match query_fasta_reader.fetch_seq(query_name, query_start, query_end - 1) {
                Ok(seq) => {
                    let mut rc = reverse_complement(&seq.to_vec());
                    unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                    rc.iter_mut()
                        .for_each(|byte| *byte = byte.to_ascii_uppercase());
                    rc
                }
                Err(e) => {
                    error!("Failed to fetch query sequence: {}", e);
                    std::process::exit(1);
                }
            }
        };

        // Fetch target sequence from target FASTA
        let target_seq = match target_fasta_reader.fetch_seq(
            target_name,
            target_start,
            target_end - 1,
        ) {
            Ok(seq) => {
                let mut seq_vec = seq.to_vec();
                unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                seq_vec
                    .iter_mut()
                    .for_each(|byte| *byte = byte.to_ascii_uppercase());
                seq_vec
            }
            Err(e) => {
                error!("Failed to fetch target sequence: {}", e);
                std::process::exit(1);
            }
        };

        // Create thread-local aligner
        // let mut aligner = AffineWavefronts::with_penalties_affine2p(
        //     0, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2,
        // );
        // let realn_cigar = align_sequences_wfa(&query_seq, &target_seq, &mut aligner);
        // let realn_cigar = cigar_ops_to_cigar_string(&realn_cigar);
        // // let paf_cigar = &realn_cigar;

        // // Convert CIGAR to tracepoints using query (A) and target (B) coordinates.
        // let tracepoints = cigar_to_tracepoints(&paf_cigar, max_diff);
        // let variable_tracepoints = cigar_to_variable_tracepoints(paf_cigar, max_diff);

        // // Compare tracepoints (allowing variable tracepoints to have None for second coordinate)
        // if tracepoints
        //     .iter()
        //     .zip(variable_tracepoints.iter())
        //     .any(|(a, b)| {
        //         a.0 != b.0 || (b.1.is_some() && Some(a.1) != b.1)
        //     })
        // {
        //     println!("Tracepoints mismatch! {}", line);
        //     println!("\t         tracepoints: {:?}", tracepoints);
        //     println!("\tvariable_tracepoints: {:?}", variable_tracepoints);
        //     std::process::exit(1);
        // }

        // // Reconstruct the CIGAR from tracepoints.
        // let cigar_from_tracepoints = tracepoints_to_cigar(
        //     &tracepoints,
        //     &query_seq,
        //     &target_seq,
        //     0,
        //     0,
        //     (mismatch,
        //     gap_open1,
        //     gap_ext1,
        //     gap_open2,
        //     gap_ext2),
        // );
        // let cigar_from_variable_tracepoints = variable_tracepoints_to_cigar(
        //     &variable_tracepoints,
        //     &query_seq,
        //     &target_seq,
        //     0,
        //     0,
        //     (mismatch,
        //     gap_open1,
        //     gap_ext1,
        //     gap_open2,
        //     gap_ext2),
        // );

        // let (matches, mismatches, insertions, inserted_bp, deletions, deleted_bp, paf_gap_compressed_id, paf_block_id) = calculate_cigar_stats(&paf_cigar);
        // let (tracepoints_matches, tracepoints_mismatches, tracepoints_insertions, tracepoints_inserted_bp, tracepoints_deletions, tracepoints_deleted_bp, tracepoints_gap_compressed_id, tracepoints_block_id) = calculate_cigar_stats(&cigar_from_tracepoints);
        // let (variable_tracepoints_matches, variable_tracepoints_mismatches, variable_tracepoints_insertions, variable_tracepoints_inserted_bp, variable_tracepoints_deletions, variable_tracepoints_deleted_bp, variable_tracepoints_gap_compressed_id, variable_tracepoints_block_id) = calculate_cigar_stats(&cigar_from_variable_tracepoints);
        // let (realign_matches, realign_mismatches, realign_insertions, realign_inserted_bp, realign_deletions, realign_deleted_bp, realign_gap_compressed_id, realign_block_id) = calculate_cigar_stats(&realn_cigar);

        // let score_from_realign = compute_alignment_score_from_cigar(&realn_cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
        // let score_from_paf = compute_alignment_score_from_cigar(&paf_cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
        // let score_from_tracepoints = compute_alignment_score_from_cigar(&cigar_from_tracepoints, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
        // let score_from_variable_tracepoints = compute_alignment_score_from_cigar(&cigar_from_variable_tracepoints, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);

        // if cigar_from_tracepoints != cigar_from_variable_tracepoints || (paf_cigar != cigar_from_tracepoints && score_from_paf != score_from_tracepoints) //&& paf_gap_compressed_id != tracepoints_gap_compressed_id
        // {
        //     println!("CIGAR mismatch! {}", line);
        //     println!("\t seqa: {}", String::from_utf8(query_seq.clone()).unwrap());
        //     println!("\t seqb: {}", String::from_utf8(target_seq.clone()).unwrap());
        //     println!("\t             CIGAR from realign: {}", realn_cigar);
        //     println!("\t                 CIGAR from PAF: {}", paf_cigar);
        //     println!("\t         CIGAR from tracepoints: {}", cigar_from_tracepoints);
        //     println!("\tCIGAR from variable_tracepoints: {}", cigar_from_variable_tracepoints);
        //     println!("\t             CIGAR score from realign: {}", score_from_realign);
        //     println!("\t                 CIGAR score from PAF: {}", score_from_paf);
        //     println!("\t         CIGAR score from tracepoints: {}", score_from_tracepoints);
        //     println!("\tCIGAR score from variable tracepoints: {}", score_from_variable_tracepoints);
        //     println!("\t              cigar stats from realign: matches: {}, mismatches: {}, insertions: {}, inserted_bp: {}, deletions: {}, deleted_bp: {}, gap_compressed_id: {:.12}, block_id: {:.12}",
        //         realign_matches, realign_mismatches, realign_insertions, realign_inserted_bp, realign_deletions, realign_deleted_bp, realign_gap_compressed_id, realign_block_id);
        //     println!("\t                  cigar stats from PAF: matches: {}, mismatches: {}, insertions: {}, inserted_bp: {}, deletions: {}, deleted_bp: {}, gap_compressed_id: {:.12}, block_id: {:.12}",
        //         matches, mismatches, insertions, inserted_bp, deletions, deleted_bp, paf_gap_compressed_id, paf_block_id);
        //     println!("\t cigar stats from          tracepoints: matches: {}, mismatches: {}, insertions: {}, inserted_bp: {}, deletions: {}, deleted_bp: {}, gap_compressed_id: {:.12}, block_id: {:.12}",
        //         tracepoints_matches, tracepoints_mismatches, tracepoints_insertions, tracepoints_inserted_bp, tracepoints_deletions, tracepoints_deleted_bp, tracepoints_gap_compressed_id, tracepoints_block_id);
        //     println!("\t cigar stats from variable_tracepoints: matches: {}, mismatches: {}, insertions: {}, inserted_bp: {}, deletions: {}, deleted_bp: {}, gap_compressed_id: {:.12}, block_id: {:.12}",
        //         variable_tracepoints_matches, variable_tracepoints_mismatches, variable_tracepoints_insertions, variable_tracepoints_inserted_bp, variable_tracepoints_deletions, variable_tracepoints_deleted_bp, variable_tracepoints_gap_compressed_id, variable_tracepoints_block_id);
        //     println!("\t         tracepoints: {:?}", tracepoints);
        //     println!("\tvariable_tracepoints: {:?}", variable_tracepoints);
        //     println!("\t                 bounds CIGAR from PAF: {:?}", get_cigar_diagonal_bounds(&paf_cigar));
        //     println!("\t         bounds CIGAR from tracepoints: {:?}", get_cigar_diagonal_bounds(&cigar_from_tracepoints));
        //     println!("\tbounds CIGAR from variable_tracepoints: {:?}", get_cigar_diagonal_bounds(&cigar_from_variable_tracepoints));

        //     let (deviation, d_min, d_max, max_gap) = compute_deviation(&cigar_from_tracepoints);
        //     println!("\t                 deviation CIGAR from PAF: {:?}", compute_deviation(&paf_cigar));
        //     println!("\t         deviation CIGAR from tracepoints: {:?}", (deviation, d_min, d_max, max_gap));
        //     println!("\tdeviation CIGAR from variable_tracepoints: {:?}", compute_deviation(&cigar_from_variable_tracepoints));
        //     // println!("=> Try using --wfa-heuristic=banded-static --wfa-heuristic-parameters=-{},{}\n", std::cmp::max(max_gap, -d_min), std::cmp::max(max_gap, d_max));
        // }
        let mut c = CigarPosition {
            apos: query_start as i64,
            bpos: target_start as i64,
            cptr: 0,
            len: 0,
        };
        let mut bundle = TPBundle {
            diff: 0,
            tlen: 0,
            trace: Vec::new(),
        };
        // lib_tracepoints::cigar2tp(&mut c, paf_cigar, aend, bend, 100 as i64, &mut bundle);

        // Convert bundle.trace (Vec<i64>) to Vec<(usize, usize)>
        //let mut tp_vec = Vec::new();
        //let trace = &bundle.trace;
        //let mut i = 0;
        //while i + 1 < trace.len() {
        //    tp_vec.push((trace[i] as usize, trace[i + 1] as usize));
        //    i += 2;
        //}
        let original_score = compute_edit_distance_alignment_score_from_cigar(&paf_cigar).unwrap_or(0);

        //format_tracepoints(&tp_vec);
        println!("Query: {}:{}-{} | Edit Distance Score: {}", 
         query_name, query_start, query_end, original_score);
        //println!("Target chunk: {}", String::from_utf8_lossy(&target_seq));
        //println!("Query chunk: {}", String::from_utf8_lossy(&query_seq));
        // println!("Tracepoints: {:?}", tp_vec);
    });
}

/// Process a chunk of lines in parallel for debugging with two PAF files
#[cfg(debug_assertions)]
fn process_debug_chunk_compare(
    lines1: &[String],
    lines2: &[String],
    query_fasta_path: &str,
    target_fasta_path: &str,
    mismatch: i32,
    gap_open1: i32,
    gap_ext1: i32,
    gap_open2: i32,
    gap_ext2: i32,
    max_diff: usize,
) {
    use std::collections::HashMap;
    
    // Parse both PAF files into HashMaps for easy comparison
    let mut paf1_records: HashMap<String, PafRecord> = HashMap::new();
    let mut paf2_records: HashMap<String, PafRecord> = HashMap::new();
    
    // Parse first PAF file
    for line in lines1 {
        if let Some(record) = parse_paf_line(line) {
            let key = create_alignment_key(&record);
            paf1_records.insert(key, record);
        }
    }
    
    // Parse second PAF file
    for line in lines2 {
        if let Some(record) = parse_paf_line(line) {
            let key = create_alignment_key(&record);
            paf2_records.insert(key, record);
        }
    }
    
    // Find common alignments and compare scores
    let common_keys: Vec<_> = paf1_records.keys()
        .filter(|k| paf2_records.contains_key(*k))
        .cloned()
        .collect();
    
    println!("Comparing {} common alignments between PAF files", common_keys.len());
    println!("PAF1 unique: {}, PAF2 unique: {}", 
             paf1_records.len() - common_keys.len(),
             paf2_records.len() - common_keys.len());
    
    // Process common alignments in parallel
    common_keys.par_iter().for_each(|key| {
        let record1 = &paf1_records[key];
        let record2 = &paf2_records[key];
        
        // Calculate scores for both records
        let score1 = compute_alignment_score_from_cigar(
            &record1.cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2
        );
        let score2 = compute_alignment_score_from_cigar(
            &record2.cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2
        );
        
        let edit_score1 = compute_edit_distance_alignment_score_from_cigar(&record1.cigar).unwrap_or(0);
        let edit_score2 = compute_edit_distance_alignment_score_from_cigar(&record2.cigar).unwrap_or(0);
        
        // Compare scores and report differences
        if score1 != score2 || edit_score1 != edit_score2 {
            println!("SCORE DIFFERENCE FOUND:");
            println!("  Alignment: {}:{}-{} {} {}:{}-{}", 
                     record1.query_name, record1.query_start, record1.query_end,
                     record1.strand, record1.target_name, record1.target_start, record1.target_end);
            println!("  PAF1 CIGAR: {}", record1.cigar);
            println!("  PAF2 CIGAR: {}", record2.cigar);
            println!("  PAF1 Alignment Score: {} | Edit Distance: {}", score1, edit_score1);
            println!("  PAF2 Alignment Score: {} | Edit Distance: {}", score2, edit_score2);
            println!("  Score Difference: {} | Edit Distance Difference: {}", 
                     score2 - score1, edit_score2 - edit_score1);
            
            // Calculate detailed statistics for both CIGARs
            let stats1 = calculate_cigar_stats(&record1.cigar);
            let stats2 = calculate_cigar_stats(&record2.cigar);
            
            println!("  PAF1 Stats: M:{} X:{} I:{} D:{} Gap-ID:{:.4} Block-ID:{:.4}", 
                     stats1.0, stats1.1, stats1.2, stats1.4, stats1.6, stats1.7);
            println!("  PAF2 Stats: M:{} X:{} I:{} D:{} Gap-ID:{:.4} Block-ID:{:.4}", 
                     stats2.0, stats2.1, stats2.2, stats2.4, stats2.6, stats2.7);
            println!();
        }
    });
    
    // Report summary statistics
    let total_score_diff: i32 = common_keys.par_iter().map(|key| {
        let record1 = &paf1_records[key];
        let record2 = &paf2_records[key];
        let score1 = compute_alignment_score_from_cigar(
            &record1.cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2
        );
        let score2 = compute_alignment_score_from_cigar(
            &record2.cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2
        );
        (score2 - score1).abs()
    }).sum();
    
    println!("SUMMARY:");
    println!("  Total alignments compared: {}", common_keys.len());
    println!("  Total absolute score difference: {}", total_score_diff);
    println!("  Average score difference per alignment: {:.2}", 
             total_score_diff as f64 / common_keys.len() as f64);
}

/// Parse a PAF line into a PafRecord
#[cfg(debug_assertions)]
fn parse_paf_line(line: &str) -> Option<PafRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return None;
    }
    
    let cigar_field = fields.iter().find(|&&s| s.starts_with("cg:Z:"))?;
    let cigar = cigar_field[5..].to_string();
    
    Some(PafRecord {
        query_name: fields[0].to_string(),
        query_start: fields[2].parse().ok()?,
        query_end: fields[3].parse().ok()?,
        strand: fields[4].to_string(),
        target_name: fields[5].to_string(),
        target_start: fields[7].parse().ok()?,
        target_end: fields[8].parse().ok()?,
        cigar,
        original_line: line.to_string(),
    })
}

/// Create a unique key for alignment comparison
#[cfg(debug_assertions)]
fn create_alignment_key(record: &PafRecord) -> String {
    format!("{}:{}:{}:{}:{}:{}:{}", 
            record.query_name, record.query_start, record.query_end,
            record.strand, record.target_name, record.target_start, record.target_end)
}

/// Updated main function debug section to handle two PAF files
#[cfg(debug_assertions)]
fn process_two_paf_files(
    paf1_path: &str,
    paf2_path: &str,
    query_fasta_path: &str,
    target_fasta_path: &str,
    mismatch: i32,
    gap_open1: i32,
    gap_ext1: i32,
    gap_open2: i32,
    gap_ext2: i32,
    max_diff: usize,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set the thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    
    // Read both PAF files
    let paf1_reader = get_paf_reader(paf1_path)?;
    let paf2_reader = get_paf_reader(paf2_path)?;
    
    let mut paf1_lines = Vec::new();
    let mut paf2_lines = Vec::new();
    
    // Read all lines from first PAF
    for line_result in paf1_reader.lines() {
        let line = line_result?;
        if !line.trim().is_empty() && !line.starts_with('#') {
            paf1_lines.push(line);
        }
    }
    
    // Read all lines from second PAF
    for line_result in paf2_reader.lines() {
        let line = line_result?;
        if !line.trim().is_empty() && !line.starts_with('#') {
            paf2_lines.push(line);
        }
    }
    
    println!("PAF1 records: {}", paf1_lines.len());
    println!("PAF2 records: {}", paf2_lines.len());
    
    // Process comparison
    process_debug_chunk_compare(
        &paf1_lines,
        &paf2_lines,
        query_fasta_path,
        target_fasta_path,
        mismatch,
        gap_open1,
        gap_ext1,
        gap_open2,
        gap_ext2,
        max_diff,
    );
    
    Ok(())
}

/// Calculate gap compressed identity and block identity from a CIGAR string
#[cfg(debug_assertions)]
fn calculate_cigar_stats(cigar: &str) -> (usize, usize, usize, usize, usize, usize, f64, f64) {
    let mut matches = 0;
    let mut mismatches = 0;
    let mut insertions = 0; // Number of insertion events
    let mut inserted_bp = 0; // Total inserted base pairs
    let mut deletions = 0; // Number of deletion events
    let mut deleted_bp = 0; // Total deleted base pairs

    // Parse CIGAR string
    let mut num_buffer = String::new();

    for c in cigar.chars() {
        if c.is_digit(10) {
            num_buffer.push(c);
        } else {
            // Get the count
            let len = num_buffer.parse::<usize>().unwrap_or(0);
            num_buffer.clear();

            match c {
                'M' => {
                    // Assuming 'M' represents matches for simplicity (as in your code)
                    matches += len;
                }
                '=' => {
                    matches += len;
                }
                'X' => {
                    mismatches += len;
                }
                'I' => {
                    insertions += 1; // One insertion event
                    inserted_bp += len; // Total inserted bases
                }
                'D' => {
                    deletions += 1; // One deletion event
                    deleted_bp += len; // Total deleted bases
                }
                'S' | 'H' | 'P' | 'N' => {
                    // Skip soft clips, hard clips, padding, and skipped regions
                }
                _ => {
                    // Unknown operation, skip
                }
            }
        }
    }

    // Calculate gap compressed identity
    let gap_compressed_identity = if matches + mismatches + insertions + deletions > 0 {
        (matches as f64) / (matches + mismatches + insertions + deletions) as f64
    } else {
        0.0
    };

    // Calculate block identity
    let edit_distance = mismatches + inserted_bp + deleted_bp;
    let block_identity = if matches + edit_distance > 0 {
        (matches as f64) / (matches + edit_distance) as f64
    } else {
        0.0
    };

    (
        matches,
        mismatches,
        insertions,
        inserted_bp,
        deletions,
        deleted_bp,
        gap_compressed_identity,
        block_identity,
    )
}

#[cfg(debug_assertions)]
fn compute_alignment_score_from_cigar(
    cigar: &str,
    mismatch: i32,
    gap_open1: i32,
    gap_ext1: i32,
    gap_open2: i32,
    gap_ext2: i32,
) -> i32 {
    let mut score = 0i32;
    let mut num_buffer = String::new();

    for c in cigar.chars() {
        if c.is_digit(10) {
            num_buffer.push(c);
        } else {
            // Get the count
            let len = num_buffer.parse::<i32>().unwrap_or(0);
            num_buffer.clear();

            match c {
                '=' => {
                    // Matches - no penalty (score 0)
                    // score += 0;
                }
                'M' => {
                    // For 'M' operations, we'd need the actual sequences to determine matches vs mismatches
                    // For now, we'll treat 'M' as matches (you may want to adjust this)
                    // score += 0;
                    eprintln!(
                        "Warning: 'M' in CIGAR requires sequences to determine match/mismatch"
                    );
                }
                'X' => {
                    // Mismatches
                    score -= mismatch * len;
                }
                'I' | 'D' => {
                    // Gaps - using dual affine model
                    // Calculate both penalty options and take the minimum (best score)
                    let score1 = gap_open1 + gap_ext1 * len;
                    let score2 = gap_open2 + gap_ext2 * len;
                    let gap_penalty = std::cmp::min(score1, score2);
                    score -= gap_penalty;
                }
                'S' | 'H' | 'P' | 'N' => {
                    // Soft clips, hard clips, padding, and skipped regions
                    // These typically don't contribute to the alignment score
                }
                _ => {
                    eprintln!("Unknown CIGAR operation: {}", c);
                }
            }
        }
    }

    score
}

#[cfg(debug_assertions)]
fn compute_edit_distance_alignment_score_from_cigar(
    cigar_string: &str
) -> Result<i32, String> {
    let mut score = 0i32;
    let mut i = 0;
    let chars: Vec<char> = cigar_string.chars().collect();
    
    while i < chars.len() {
        // Parse the length (number before the operation)
        let mut length_str = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            length_str.push(chars[i]);
            i += 1;
        }
        
        // Check if we have a valid length
        if length_str.is_empty() {
            return Err(format!("Invalid CIGAR format: expected length before operation at position {}", i));
        }
        
        let length: u32 = length_str.parse()
            .map_err(|_| format!("Invalid length in CIGAR: '{}'", length_str))?;
        
        // Parse the operation
        if i >= chars.len() {
            return Err("Invalid CIGAR format: expected operation after length".to_string());
        }
        
        let operation = chars[i];
        i += 1;
        
        // Add to score based on operation
        match operation {
            'M' | '=' => {
                // Match/Exact Match: no penalty
                // Note: 'M' can be ambiguous (match or mismatch) but in edit distance context,
                // it's typically treated as a match with 0 penalty
            }
            'X' | 'D' | 'I' => {
                // Mismatch, Deletion, Insertion: +length penalty
                score += length as i32;
            }
            _ => {
                return Err(format!("Unknown CIGAR operation: '{}'", operation));
            }
        }
    }
    
    Ok(score)
}

/// Initialize logger based on verbosity
fn setup_logger(verbosity: u8) {
    env_logger::Builder::new()
        .filter_level(match verbosity {
            0 => log::LevelFilter::Warn,  // Errors and warnings
            1 => log::LevelFilter::Info,  // Errors, warnings, and info
            _ => log::LevelFilter::Debug, // Errors, warnings, info, and debug
        })
        .init();
}

fn get_paf_reader(paf: &str) -> io::Result<Box<dyn BufRead>> {
    if paf == "-" {
        Ok(Box::new(BufReader::new(std::io::stdin())))
    } else if paf.ends_with(".gz") || paf.ends_with(".bgz") {
        let file = File::open(paf)?;
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        let file = File::open(paf)?;
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Reverse a CIGAR string by reversing the order of operations
/// This is used when the alignment is on the negative strand
fn reverse_cigar(cigar: &str) -> String {
    // Parse CIGAR into operations
    let operations = parse_cigar_to_operations(cigar);
    
    // Reverse the order of operations
    let reversed_ops: Vec<_> = operations.into_iter().rev().collect();
    
    // Convert back to CIGAR string
    operations_to_cigar_string(&reversed_ops)
}

/// Helper function to parse CIGAR string into (length, operation) pairs
fn parse_cigar_to_operations(cigar: &str) -> Vec<(u32, char)> {
    let mut operations = Vec::new();
    let mut num_buffer = String::new();
    
    for c in cigar.chars() {
        if c.is_ascii_digit() {
            num_buffer.push(c);
        } else {
            let length = num_buffer.parse::<u32>().unwrap_or(1);
            operations.push((length, c));
            num_buffer.clear();
        }
    }
    
    operations
}

/// Helper function to convert operations back to CIGAR string
fn operations_to_cigar_string(operations: &[(u32, char)]) -> String {
    operations
        .iter()
        .map(|(len, op)| format!("{}{}", len, op))
        .collect::<String>()
}

/// Process a chunk of lines in parallel for compression
fn process_compress_chunk(lines: &[String], mixed: bool, variable: bool, max_diff: usize) {
    lines.par_iter().for_each(|line| {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping malformed PAF line", line)
            );
            std::process::exit(1);
        }

        let Some(cg_field) = fields.iter().find(|&&s| s.starts_with("cg:Z:")) else {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping CIGAR-less PAF line", line)
            );
            std::process::exit(1);
        };
        let raw_cigar = &cg_field[5..];

        // Parse PAF fields for coordinate transformation
        let query_start: usize = fields[2].parse().unwrap_or_else(|_| {
            error!("Invalid query_start in PAF line");
            std::process::exit(1);
        });
        let query_end: usize = fields[3].parse().unwrap_or_else(|_| {
            error!("Invalid query_end in PAF line");
            std::process::exit(1);
        });
        let strand = fields[4];
        let target_len: usize = fields[6].parse().unwrap_or_else(|_| {
            error!("Invalid target_len in PAF line");
            std::process::exit(1);
        });
        let paf_target_start: usize = fields[7].parse().unwrap_or_else(|_| {
            error!("Invalid target_start in PAF line");
            std::process::exit(1);
        });
        let paf_target_end: usize = fields[8].parse().unwrap_or_else(|_| {
            error!("Invalid target_end in PAF line");
            std::process::exit(1);
        });

        // Transform coordinates based on strand (similar to ALNtoPAF.c logic)
        let (target_start, target_end, cigar) = if strand == "-" {
            // For reverse complement, transform coordinates and reverse CIGAR
            let transformed_start = target_len - paf_target_end;
            let transformed_end = target_len - paf_target_start;
            let reversed_cigar = reverse_cigar(raw_cigar);
            (transformed_start, transformed_end, reversed_cigar)
        } else {
            // For forward strand, use coordinates as-is
            (paf_target_start, paf_target_end, raw_cigar.to_string())
        };

        // Convert CIGAR based on options and add type prefix
        let tracepoints_str = if mixed {
            // Use mixed representation
            let tp = cigar_to_mixed_tracepoints(&cigar, max_diff);
            format_mixed_tracepoints(&tp)
        } else if variable {
            // Use variable tracepoints
            let tp = cigar_to_variable_tracepoints(&cigar, max_diff);
            format_variable_tracepoints(&tp)
        } else {
            // Use cigar2tp for standard tracepoints
            let mut c = CigarPosition {
                apos: query_start as i64,
                bpos: target_start as i64,  // Use transformed coordinate
                cptr: 0,
                len: 0,
            };
            let mut bundle = TPBundle {
                diff: 0,
                tlen: 0,
                trace: Vec::new(),
            };
            lib_tracepoints::cigar2tp(
                &mut c, 
                &cigar, 
                query_end as i64, 
                target_end as i64,  // Use transformed coordinate
                100 as i64, 
                &mut bundle
            );

            // Convert bundle.trace to Vec<(usize, usize)>
            let mut tp_vec = Vec::new();
            let trace = &bundle.trace;
            let mut i = 0;
            while i + 1 < trace.len() {
                tp_vec.push((trace[i] as usize, trace[i + 1] as usize));
                i += 2;
            }
            format_tracepoints(&tp_vec)
        };

        // Print the result
        let new_line = line.replace(cg_field, &format!("tp:Z:{}", tracepoints_str));
        println!("{}", new_line);
    });
}

/// Process a chunk of lines in parallel for decompression
fn process_decompress_chunk(
    lines: &[String],
    tp_type: &TracepointType,
    query_fasta_path: &str,
    target_fasta_path: &str,
    mismatch: i32,
    gap_open1: i32,
    gap_ext1: i32,
    gap_open2: i32,
    gap_ext2: i32,
) {
    lines.par_iter().for_each(|line| {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping malformed PAF line", line)
            );
            std::process::exit(1);
        }

        let Some(tp_field) = fields.iter().find(|&&s| s.starts_with("tp:Z:")) else {
            error!(
                "{}",
                message_with_truncate_paf_file("Skipping tracepoints-less PAF line", line)
            );
            std::process::exit(1);
        };
        let tracepoints_str = &tp_field[5..];

        // Parse mandatory PAF fields
        let query_name = fields[0];
        let query_start: usize = fields[2].parse().unwrap_or_else(|_| {
            error!(
                "{}",
                message_with_truncate_paf_file("Invalid query_start in PAF line", line)
            );
            std::process::exit(1);
        });
        let query_end: usize = fields[3].parse().unwrap_or_else(|_| {
            error!(
                "{}",
                message_with_truncate_paf_file("Invalid query_end in PAF line", line)
            );
            std::process::exit(1);
        });
        let strand = fields[4];
        let target_name = fields[5];
        let target_len: usize = fields[6].parse().unwrap_or_else(|_| {
            error!(
                "{}",
                message_with_truncate_paf_file("Invalid target_len in PAF line", line)
            );
            std::process::exit(1);
        });
        let paf_target_start: usize = fields[7].parse().unwrap_or_else(|_| {
            error!(
                "{}",
                message_with_truncate_paf_file("Invalid target_start in PAF line", line)
            );
            std::process::exit(1);
        });
        let paf_target_end: usize = fields[8].parse().unwrap_or_else(|_| {
            error!(
                "{}",
                message_with_truncate_paf_file("Invalid target_end in PAF line", line)
            );
            std::process::exit(1);
        });

        // Create thread-local FASTA readers
        let query_fasta_reader = FastaReader::from_path(query_fasta_path).unwrap_or_else(|e| {
            error!("Failed to create query FASTA reader: {}", e);
            std::process::exit(1);
        });

        let target_fasta_reader = FastaReader::from_path(target_fasta_path).unwrap_or_else(|e| {
            error!("Failed to create target FASTA reader: {}", e);
            std::process::exit(1);
        });

        // Fetch query sequence (always forward strand for the query)
        let query_seq = match query_fasta_reader.fetch_seq(query_name, query_start, query_end - 1) {
            Ok(seq) => {
                let mut seq_vec = seq.to_vec();
                unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                seq_vec
                    .iter_mut()
                    .for_each(|byte| *byte = byte.to_ascii_uppercase());
                seq_vec
            }
            Err(e) => {
                error!("Failed to fetch query sequence: {}", e);
                std::process::exit(1);
            }
        };

        // Calculate target coordinates and fetch target sequence based on strand
        // This mirrors the logic in ALNtoPAF.c lines 234-240
        let (target_seq, target_coord_start, target_coord_end) = if strand == "-" {
            // For reverse complement alignments:
            // 1. Transform coordinates (similar to ALNtoPAF.c bmin/bmax calculation)
            let coord_start = paf_target_start;
            let coord_end = paf_target_end;
            
            // 2. Fetch the sequence from the transformed coordinates
            let seq = match target_fasta_reader.fetch_seq(target_name, coord_start, coord_end - 1) {
                Ok(seq) => {
                    let mut seq_vec = seq.to_vec();
                    unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                    seq_vec
                        .iter_mut()
                        .for_each(|byte| *byte = byte.to_ascii_uppercase());
                    seq_vec
                }
                Err(e) => {
                    error!("Failed to fetch target sequence: {}", e);
                    std::process::exit(1);
                }
            };
            
            // 3. Reverse complement the sequence (similar to Complement_Seq in ALNtoPAF.c)
            let rc_seq = reverse_complement(&seq);
            
            (rc_seq, coord_start, coord_end)
        } else {
            // For forward strand, use coordinates as-is
            let seq = match target_fasta_reader.fetch_seq(target_name, paf_target_start, paf_target_end - 1) {
                Ok(seq) => {
                    let mut seq_vec = seq.to_vec();
                    unsafe { libc::free(seq.as_ptr() as *mut std::ffi::c_void) };
                    seq_vec
                        .iter_mut()
                        .for_each(|byte| *byte = byte.to_ascii_uppercase());
                    seq_vec
                }
                Err(e) => {
                    error!("Failed to fetch target sequence: {}", e);
                    std::process::exit(1);
                }
            };
            
            (seq, paf_target_start, paf_target_end)
        };

        // Convert tracepoints back to CIGAR using the properly oriented sequences
        let reconstructed_cigar = match tp_type {
            TracepointType::Mixed => {
                let mixed_tracepoints = parse_mixed_tracepoints(tracepoints_str);
                mixed_tracepoints_to_cigar(
                    &mixed_tracepoints,
                    &query_seq,
                    &target_seq,
                    0,  // Use 0-based coordinates since sequences are already extracted
                    0,
                    (mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2),
                )
            }
            TracepointType::Variable => {
                let variable_tracepoints = parse_variable_tracepoints(tracepoints_str);
                variable_tracepoints_to_cigar(
                    &variable_tracepoints,
                    &query_seq,
                    &target_seq,
                    0,
                    0,
                    (mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2),
                )
            }
            TracepointType::Standard => {
                let tracepoints = parse_tracepoints(tracepoints_str);
                tracepoints_to_fastga_cigar(
                    &tracepoints,
                    &query_seq,
                    &target_seq,
                    0,
                    0,
                    query_start,
                    (0, 1, 1),
                )
            }
        };

        // For reverse complement alignments, reverse the CIGAR back to match PAF format
        // This mirrors the CIGAR reversal logic in ALNtoPAF.c lines 415-422
        let final_cigar = if strand == "--" {
            reverse_cigar(&reconstructed_cigar)
        } else {
            reconstructed_cigar
        };

        // Replace tracepoints with CIGAR
        let new_line = line.replace(tp_field, &format!("cg:Z:{}", final_cigar));
        println!("{}", new_line);
    });
}

/// Combines a message with the first 9 columns of a PAF line.
fn message_with_truncate_paf_file(message: &str, line: &str) -> String {
    let truncated_line = line.split('\t').take(9).collect::<Vec<&str>>().join("\t");
    format!("{}: {} ...", message, truncated_line)
}

fn format_tracepoints(tracepoints: &[(usize, usize)]) -> String {
    tracepoints
        .iter()
        .map(|(a, b)| format!("{},{}", a, b))
        .collect::<Vec<String>>()
        .join(";")
}
fn format_mixed_tracepoints(mixed_tracepoints: &[MixedRepresentation]) -> String {
    mixed_tracepoints
        .iter()
        .map(|tp| match tp {
            MixedRepresentation::Tracepoint(a, b) => format!("{},{}", a, b),
            MixedRepresentation::CigarOp(len, op) => format!("{}{}", len, op),
        })
        .collect::<Vec<String>>()
        .join(";")
}

fn format_variable_tracepoints(variable_tracepoints: &[(usize, Option<usize>)]) -> String {
    variable_tracepoints
        .iter()
        .map(|(a, b_opt)| match b_opt {
            Some(b) => format!("{},{}", a, b),
            None => format!("{}", a),
        })
        .collect::<Vec<String>>()
        .join(";")
}
fn parse_penalties(
    penalties: &str,
) -> Result<(i32, i32, i32, i32, i32), Box<dyn std::error::Error>> {
    let tokens: Vec<&str> = penalties.split(',').collect();
    if tokens.len() != 5 {
        error!(
            "Error: penalties must be provided as mismatch,gap_open1,gap_ext1,gap_open2,gap_ext2"
        );
        std::process::exit(1);
    }

    let mismatch: i32 = tokens[0].parse()?;
    let gap_open1: i32 = tokens[1].parse()?;
    let gap_ext1: i32 = tokens[2].parse()?;
    let gap_open2: i32 = tokens[3].parse()?;
    let gap_ext2: i32 = tokens[4].parse()?;

    Ok((mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2))
}

fn parse_tracepoints(tp_str: &str) -> Vec<(usize, usize)> {
    tp_str
        .split(';')
        .filter_map(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            Some((parts[0].parse().unwrap(), parts[1].parse().unwrap()))
        })
        .collect()
}

fn parse_mixed_tracepoints(tp_str: &str) -> Vec<MixedRepresentation> {
    tp_str
        .split(';')
        .filter_map(|s| {
            if s.contains(',') {
                // This is a tracepoint
                let parts: Vec<&str> = s.split(',').collect();
                Some(MixedRepresentation::Tracepoint(
                    parts[0].parse().unwrap(),
                    parts[1].parse().unwrap(),
                ))
            } else {
                // This is a cigar operation
                let mut chars = s.chars();
                let mut len_str = String::new();

                // Read digits
                while let Some(c) = chars.next() {
                    if c.is_digit(10) {
                        len_str.push(c);
                    } else {
                        // Found operator character
                        let len = len_str.parse().unwrap();
                        return Some(MixedRepresentation::CigarOp(len, c));
                    }
                }

                // If we get here, parsing failed
                None
            }
        })
        .collect()
}

fn parse_variable_tracepoints(tp_str: &str) -> Vec<(usize, Option<usize>)> {
    tp_str
        .split(';')
        .filter_map(|s| {
            if s.contains(',') {
                // This has both coordinates
                let parts: Vec<&str> = s.split(',').collect();
                Some((parts[0].parse().unwrap(), Some(parts[1].parse().unwrap())))
            } else {
                // This has only first coordinate
                match s.parse() {
                    Ok(a) => Some((a, None)),
                    Err(_) => None,
                }
            }
        })
        .collect()
}

#[cfg(debug_assertions)]
fn get_cigar_diagonal_bounds(cigar: &str) -> (i64, i64) {
    let mut current_diagonal = 0; // Current diagonal position
    let mut min_diagonal = 0; // Lowest diagonal reached
    let mut max_diagonal = 0; // Highest diagonal reached

    // Parse CIGAR string with numerical counts
    let mut num_buffer = String::new();

    for c in cigar.chars() {
        if c.is_digit(10) {
            num_buffer.push(c);
        } else {
            // Get the count
            let count = num_buffer.parse::<i64>().unwrap();
            num_buffer.clear();

            match c {
                'M' | '=' | 'X' => {
                    // Matches stay on same diagonal
                }
                'D' => {
                    // Deletions move down diagonal by count amount
                    current_diagonal -= count;
                    min_diagonal = min_diagonal.min(current_diagonal);
                }
                'I' => {
                    // Insertions move up diagonal by count amount
                    current_diagonal += count;
                    max_diagonal = max_diagonal.max(current_diagonal);
                }
                _ => panic!("Invalid CIGAR operation: {}", c),
            }
        }
    }

    (min_diagonal, max_diagonal)
}

#[cfg(debug_assertions)]
fn compute_deviation(cigar: &str) -> (i64, i64, i64, i64) {
    let mut deviation = 0;
    let mut d_max = -10000;
    let mut d_min = 10000;
    let mut max_gap = 0;

    // Parse CIGAR string with numerical counts
    let mut num_buffer = String::new();

    for c in cigar.chars() {
        if c.is_digit(10) {
            num_buffer.push(c);
        } else {
            // Get the count
            let count = num_buffer.parse::<i64>().unwrap();
            num_buffer.clear();

            match c {
                'M' | '=' | 'X' => {
                    // Matches stay on same diagonal
                }
                'D' => {
                    // Deletions move down diagonal by count amount
                    deviation -= count;
                    max_gap = std::cmp::max(max_gap, count);
                }
                'I' => {
                    deviation += count;
                    max_gap = std::cmp::max(max_gap, count);
                }
                _ => panic!("Invalid CIGAR operation: {}", c),
            }

            d_max = std::cmp::max(d_max, deviation);
            d_min = std::cmp::min(d_min, deviation);
        }
    }

    (deviation, d_min, d_max, max_gap)
}

/// Returns the reverse complement of a DNA sequence
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&c| match c {
            b'A' => b'T',
            b'T' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            b'N' => b'N',
            _ => b'N', // Convert any unexpected bases to N
        })
        .collect()
}

// /// Calculate alignment coordinates from a CIGAR string and starting positions
// /// Returns (query_end, query_len, target_end, target_len)
// fn calculate_alignment_coordinates(
//     cigar: &str,
//     query_start: usize,
//     target_start: usize,
// ) -> (usize, usize, usize, usize) {
//     let ops = cigar_str_to_cigar_ops(cigar);

//     let mut query_len = 0;
//     let mut target_len = 0;

//     // Calculate total lengths by checking which operations consume query/target bases
//     for &(len, op) in &ops {
//         if consumes_a(op) {
//             query_len += len;
//         }
//         if consumes_b(op) {
//             target_len += len;
//         }
//     }

//     // Calculate end positions by adding consumed lengths to start positions
//     let query_end = query_start + query_len;
//     let target_end = target_start + target_len;

//     (query_end, query_len, target_end, target_len)
// }
mod tests {
    use super::*;

    #[test]
    fn test_compute_alignment_score_from_cigar() {
        // Test parameters (using default penalties: 5,8,2,24,1)
        let mismatch = 1;
        let gap_open1 = 1;
        let gap_ext1 = -1;
        let gap_open2 = -1;
        let gap_ext2 = -1;

        // Test 1: Perfect match - should have score 0
        let cigar1 = "29=1I1=1X1D12=1X5=1X35=1X5=1X1=1X6=1X8=1X1=1D1=1X1=1I6=1X17=1X17=1X5=1X4=1X3=1X5=1X2=1X14=1X2=1X4=1X3=1X1D1=1I8=2X2=1X2=1X14=1X2=1X2=1X2=1X5=1X5=1X5=1X2=1X19=1X2=2D1=1D6=1X1=1X3=1X1=1X2D2=1D2=2X6=1X5=1X8=1X2=1X33=1D2=1X4=2D3=1X4=2D1=2D2=1D3=1D2=1D6=1X4=1I5=1D3=1X1=1X3=1X2=1X1=1I6=";
        let score1 = compute_alignment_score_from_cigar(cigar1, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
	let fastga_cigar = "29=1I1=1X1D12=1X5=1X35=1X5=1X1=1X6=1X8=1I2=1X1=1X1D6=1X17=1X17=1X5=1X4=1X3=1X5=1X2=1X14=1X2=1X4=1X3=1I1=1X1D8=2X2=1X2=1X14=1X2=1X2=1X2=1X5=1X5=1X5=1X2=1X19=1X2=2X2=1X8=1X1=2X2=1X1=6D1X6=1X5=1X8=1X2=1X33=1D2=1X4=9D6=1X7=1X7=1X4=1I5=1D3=1X1=1X3=1X2=1X1=1I6=";
	let fastga_score = compute_alignment_score_from_cigar(fastga_cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
	let cigzip_cigar = "29=3X12=1X5=1X35=1X5=1I1=1I1X2=2I1D1=2D1X1=1D7=1I2=1X1=1D1X6=1X17=1X17=1X5=1X4=1X3=1X5=1X2=1X13=4I1=1D2=3D2=1X3=3X8=2X2=1X2=1X14=1X2=1X2=1X2=1X5=1X5=1X5=1X2=1X19=1X2=1I1=3I1D2=1D1=1D1=1D4=1X1=2X2=2D2=1D1X1=1X4=1D1=1D1=1D4=1X8=1X2=1X33=1D2=1X4=2D3=1X4=1X1=1D1=1D2=1X1=2X1=4D1=1D3=1X4=1I5=1D3=1X1=1X3=1X2=1X1=1I6=";
	let cigzip_score = compute_alignment_score_from_cigar(cigzip_cigar, mismatch, gap_open1, gap_ext1, gap_open2, gap_ext2);
        // assert_eq!(score1, 0, "Perfect match should have score 0");
	println!("edlib Score: {}", score1);
	println!("fastga Score: {}", fastga_score);
	println!("cigzip Score: {}", cigzip_score);
    }
}
