#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import mkdir_if_not_exist
import sys


def get_file_len(fname):
    """
    Get the number of lines in a file.
    """
    with open(fname, 'r') as f:
        return len(f.readlines())


def merge_files(ref_key_file, to_merge_files, base_dir, to_merge_dir, output_dir):
    """
    Merge the flexible alignment (sentence) results and form one stm file for each utt.
    """
    mkdir_if_not_exist(output_dir)
    ref_key = []
    with open(ref_key_file, 'r') as f:
        for line in f:
            uttid = line.strip().split(maxsplit=1)[0]
            ref_key.append(uttid)

    to_merge_files = to_merge_files.split(',')
    to_merge_fp = {}
    base_fp = {}
    for to_merge_fname in to_merge_files:
        to_merge_file = Path(to_merge_fname)
        to_merge_fp[to_merge_fname] = to_merge_dir / to_merge_file
        base_fp[to_merge_fname] = base_dir / to_merge_file

    base_len = sys.maxsize
    for file in base_fp.values():
        base_len = min(base_len, get_file_len(file))

    to_merge_len = sys.maxsize
    for file in to_merge_fp.values():
        to_merge_len = min(to_merge_len, get_file_len(file))

    # Read the base files
    base_elems = {}
    for fname in to_merge_files:
        base_elems[fname] = {}
        with open(base_fp[fname], 'r') as f:
            for line in f:
                uttid = line.strip().split(maxsplit=1)[0]
                base_elems[fname][uttid] = line.strip()

    # Read the files to merge
    to_merge_elems = {}
    for fname in to_merge_files:
        to_merge_elems[fname] = {}
        with open(to_merge_fp[fname], 'r') as f:
            for line in f:
                uttid = line.strip().split(maxsplit=1)[0]
                to_merge_elems[fname][uttid] = line.strip()

    out_fps = {}
    for fname in to_merge_files:
        out_fps[fname] = open(output_dir / fname, "w")

    for key in ref_key:
        for fname in to_merge_files:
            if key in base_elems[fname]:
                print(base_elems[fname][key], file=out_fps[fname])
            elif key in to_merge_elems[fname]:
                print(to_merge_elems[fname][key], file=out_fps[fname])

    # Close all output files
    for fname in to_merge_files:
        out_fps[fname].close()


def main():
    """
    Merge the flexible alignment (sentence) results and form one stm file for each utt.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Merge the flexible alignment (sentence) results and form one stm file for each utt.')
    parser.add_argument('--ref_key_file', type=Path, default='info',
                        help='The file containing the reference key used to sort the result.')
    parser.add_argument('--to_merge_files', type=str, required=True,
                        help='A comma delimited string containing the files to be merged.')
    parser.add_argument('--base_dir', type=Path, required=True,
                        help='The full path to the base directory with files to merge.')
    parser.add_argument('--to_merge_dir', type=Path, required=True,
                        help='The full path to the directory with with files to merge.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the output directory with merged files.')
    args = parser.parse_args()
    merge_files(ref_key_file=args.ref_key_file, to_merge_files=args.to_merge_files, base_dir=args.base_dir,
                to_merge_dir=args.to_merge_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
