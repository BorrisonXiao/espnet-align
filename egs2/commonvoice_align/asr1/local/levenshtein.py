from utils import levenshtein
import argparse
import os
from pathlib import Path


def mark_anchors(decoded_dir, text_map, output_dir):
    # Read the ground-truth text_map
    scps = {}
    with open(text_map, "r") as f:
        for line in f:
            uttid, scp_path = line.strip().split(" ", maxsplit=1)
            scps[uttid] = scp_path

    # Compute Levenshtein-distance based anchors for each utterance
    for decoded in os.listdir(decoded_dir):
        uttid = Path(decoded).stem
        if uttid not in scps:
            print(
                f"ERROR: Cannot find the {uttid}'s ground-truth text from {text_map}, either key error or the ground-truth text is not pre-processed.")
            exit(1)

        with open(os.path.join(decoded_dir, decoded), "r") as f:
            decoded_txt = f.read()
        with open(scps[uttid], "r") as f:
            scp = f.read()

        dist, d_txt, s_txt = levenshtein(decoded_txt, scp)

        with open(os.path.join(output_dir, "raw", uttid + ".anchor"), "w") as f:
            print(f"{dist} {len(decoded_txt)} {len(scp)}\n{d_txt}\n{s_txt}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Compute levenshtein distance between decoded and ground-truth text.')
    parser.add_argument('--decoded_dir', type=str, required=True,
                        help='The full path to the directory in which decoded text files are stored')
    parser.add_argument('--text_map', type=str, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which the comparison file will be stored')
    args = parser.parse_args()

    mark_anchors(args.decoded_dir, args.text_map, args.output_dir)


if __name__ == "__main__":
    main()
