from utils import Integerizer
import argparse
import os
from pathlib import Path


def mkdir_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def prep_txt(decoded_dir, text_map, output_dir, integerize=False):
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

        if integerize:
            txt_decoded, txt_scp = decoded_txt.strip().split(" "), scp.strip().split(" ")
            vocab = Integerizer(txt_decoded + txt_scp)
            decoded_txt, scp = " ".join(map(str, vocab.index(txt_decoded))), " ".join(
                map(str, vocab.index(txt_scp)))

        raw_path = os.path.join(output_dir)
        mkdir_if_not_exist(raw_path)
        with open(os.path.join(raw_path, uttid + ".decoded"), "w") as f:
            print(f"{uttid} {decoded_txt}", file=f)
        with open(os.path.join(raw_path, uttid + ".original"), "w") as f:
            print(f"{uttid} {scp}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Compute levenshtein distance between decoded and ground-truth text.')
    parser.add_argument('--decoded_dir', type=str, required=True,
                        help='The full path to the directory in which decoded text files are stored')
    parser.add_argument('--text_map', type=str, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which the comparison file will be stored')
    parser.add_argument('--integerize', action='store_true',
                        help='If option provided, the text will be integerized')
    args = parser.parse_args()

    prep_txt(args.decoded_dir, args.text_map, args.output_dir, args.integerize)


if __name__ == "__main__":
    main()
