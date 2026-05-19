#!/usr/bin/env python3
"""Generate recurrent mutation BED files from existing training/validation BED files.

This script converts standard MuRaL BED files to the recurrent mutation format
where the 4th column (name) encodes per-site information in a semicolon-separated
format: 'chrom:start;ref>alt;AC;count', where AC is the allele count (integer;
-1 if unavailable) and the last field is the observed mutation count.
"""
import random

COMPLEMENTS = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def get_ref_base(seq_dict, chrom, pos):
    """Get reference base at a position (0-based start)."""
    seq = seq_dict.get(chrom, '')
    if pos < len(seq):
        return seq[pos].upper()
    return 'N'

def generate_recurrent_bed(input_bed, output_bed, seq_dict, max_count=5, recurrent_prob=0.15):
    """Convert a standard BED file to recurrent mutation format.

    For mutated sites (score != 0), assign a random count >= 1.
    For non-mutated sites (score == 0), name stays as '.'.
    Format: chrom:start;ref>alt;AC;count  (AC set to -1 as unavailable for synthetic data)
    """
    mut_type_map_at = {1: 'A>C', 2: 'A>G', 3: 'A>T'}
    mut_type_map_cg = {1: 'C>A', 2: 'C>G', 3: 'C>T'}
    random.seed(42)

    with open(input_bed) as fin, open(output_bed, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            chrom, start, end, name, score, strand = parts
            score = int(score)

            if score == 0:
                new_name = '.'
            else:
                ref = get_ref_base(seq_dict, chrom, int(start))

                if strand == '-':
                    ref = COMPLEMENTS.get(ref, 'N')

                if ref in ('A', 'T'):
                    mut_type = mut_type_map_at.get(score, 'A>C')
                elif ref in ('C', 'G'):
                    mut_type = mut_type_map_cg.get(score, 'C>A')
                else:
                    mut_type = 'N>N'

                if random.random() < recurrent_prob:
                    count = random.randint(2, max_count)
                else:
                    count = 1

                new_name = f'{chrom}:{start};{mut_type};-1;{count}'

            fout.write(f'{chrom}\t{start}\t{end}\t{new_name}\t{score}\t{strand}\n')

def read_fasta(fasta_path):
    """Read a small FASTA file into a dict of chrom -> sequence."""
    seqs = {}
    current_chrom = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_chrom:
                    seqs[current_chrom] = ''.join(current_seq)
                current_chrom = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_chrom:
        seqs[current_chrom] = ''.join(current_seq)
    return seqs

if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snv_data_dir = os.path.join(script_dir, 'snv', 'data')

    seq_path = os.path.join(snv_data_dir, 'seq.fa')
    train_bed = os.path.join(snv_data_dir, 'training.sorted.bed')
    valid_bed = os.path.join(snv_data_dir, 'validation.sorted.bed')

    print("Reading reference genome...")
    seq_dict = read_fasta(seq_path)

    print("Generating recurrent training BED...")
    generate_recurrent_bed(
        train_bed,
        os.path.join(snv_data_dir, 'recurrent_training.sorted.bed'),
        seq_dict
    )

    print("Generating recurrent validation BED...")
    generate_recurrent_bed(
        valid_bed,
        os.path.join(snv_data_dir, 'recurrent_validation.sorted.bed'),
        seq_dict
    )

    print("Done.")
