#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
from tqdm import tqdm
from pathlib import Path
import re


def _split(input_dir, output_dir, ref_file):
    with open(ref_file, 'r') as f:
        lines = f.readlines()
        valid_spks = lines[0].split()
        test_spks = lines[1].split()
    print(f"Valid speakers: {valid_spks}")
    print(f"Test speakers: {test_spks}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the text for Levenshtein distance computation.')
    parser.add_argument('-i', '--input_dir', type=Path, required=True,
                        help='The full path to the input directory to be splitted.')
    parser.add_argument('-o', '--output_dir', type=Path, required=True,
                        help='The full path to the output directory.')
    parser.add_argument('--ref_file', type=Path, required=True,
                        help='The full path to the reference file.')
    args = parser.parse_args()

    _split(input_dir=args.input_dir, output_dir=args.output_dir, ref_file=args.ref_file)


if __name__ == "__main__":
    main()
