import argparse
import os
from pathlib import Path
from utils import scpid2mid, extract_mid, mkdir_if_not_exist
from collections import defaultdict
import logging

EPS = "***"


def read_text_map(file, decode=False):
    """
    Helper function for reading the text_map file.
    Returns a dict of {mid: [(scpid, text_fp), ...], ...}.
    """
    res = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            scpid, fp = line.strip().split(maxsplit=1)
            mid = scpid2mid(scpid) if not decode else extract_mid(scpid)
            res[mid].append((scpid, fp))

    # Sort the scps by their order in the original text
    for mid in res:
        res[mid].sort(key=lambda x: int(x[0].split("_")[2]))

    return res


def break_sents(input_dir, output_dir, text_map, decode_text_map):
    txtdir = os.path.join(output_dir, "txt")
    mkdir_if_not_exist(txtdir)

    text_map = read_text_map(text_map)
    decode_text_map = read_text_map(decode_text_map, decode=True)
    for file in os.listdir(input_dir):
        mid = Path(file).stem
        with open(os.path.join(input_dir, file), "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line == EPS:
                    continue
                start, end = list(map(int, line.split(maxsplit=1)))

                # Note: This might have some issues if the decode_text_map has missing indices (manual tests seem to pass though)
                output = os.path.join(txtdir, f"{decode_text_map[mid][i][0]}.txt")
                with open(output, "w") as out:
                    # Maybe use scpid here to keep track of the spkid
                    for scpid, fp in text_map[mid][start:end + 1]:
                        with open(fp, "r") as txt:
                            out.write(txt.read())


def main():
    parser = argparse.ArgumentParser(
        description='Break sentences for primary alignment results.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to decoded directory')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to output search map file')
    parser.add_argument('--text_map', type=Path, required=True,
                        help='The full path to sentence text_map file')
    parser.add_argument('--decode_text_map', type=Path, required=True,
                        help='The full path to the decode_text_map file')
    args = parser.parse_args()

    break_sents(input_dir=args.input_dir, output_dir=args.output_dir,
                text_map=args.text_map, decode_text_map=args.decode_text_map)


if __name__ == "__main__":
    main()
