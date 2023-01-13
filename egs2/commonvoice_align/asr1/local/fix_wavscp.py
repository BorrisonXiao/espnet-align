import argparse
import os
from pathlib import Path
import re
from utils import cut_uttid_len


def fix(input_file, output):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output, 'w') as f:
        for line in lines:
            uttid, rest = line.strip().split(maxsplit=1)
            uttid = cut_uttid_len(uttid)
            print(f'{uttid} {rest}', file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Fix filename too long issue.')
    parser.add_argument('--input_file', type=Path, required=True,
                        help='The full path to input directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to output directory')
    args = parser.parse_args()

    fix(input_file=args.input_file, output=args.output)


if __name__ == "__main__":
    main()
