#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import segid2uttid, mkdir_if_not_exist

def generate_keys(input_file, output):
    added = set()
    with open(output, "w") as ofh:
        with open(input_file, "r") as f:
            for line in f:
                segid, _ = line.strip().split(maxsplit=1)
                uttid = segid2uttid(segid)
                if uttid not in added:
                    added.add(uttid)
                    print(uttid, file=ofh)

def main():
    """
    Generate uttid keys for paralleling the clean_salign.py script.
    """
    logging.basicConfig(
        level="INFO", format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Generate uttid keys for paralleling the clean_salign.py script.')
    parser.add_argument('--input_file', type=Path, required=True,
                        help='The full path to the input file with alignments.')
    parser.add_argument('--output', type=Path, required=True,
                        help='The output keyfile.')
    args = parser.parse_args()
    generate_keys(input_file=args.input_file, output=args.output)


if __name__ == "__main__":
    main()
