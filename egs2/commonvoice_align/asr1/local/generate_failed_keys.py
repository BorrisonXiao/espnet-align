#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os


def clean_salign(ref_fp, hyp_fp, output):
    hyps = set()
    with open(hyp_fp, "r") as f:
        for line in f:
            hyps.add(line.strip().split()[0])

    with open(output, "w") as ofh:
        with open(ref_fp, "r") as f:
            for line in f:
                uttid = line.strip().split()[0]
                if uttid not in hyps:
                    print(line.strip(), file=ofh)


def main():
    """
    Add failed instances to a new file to re-run flexible alignment.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Add failed instances to re-run flexible alignment.')
    parser.add_argument('--ref_key_file', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    parser.add_argument('--hyp_key_file', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--output_key_file', type=Path, required=True,
                        help='The text_map file storing pointers to the sentence-splitted text files')
    args = parser.parse_args()
    clean_salign(ref_fp=args.ref_key_file,
                 hyp_fp=args.hyp_key_file, output=args.output_key_file)


if __name__ == "__main__":
    main()
