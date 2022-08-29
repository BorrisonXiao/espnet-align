import argparse
import os
from pathlib import Path
from utils import mkdir_if_not_exist


def generate_text(text_map, output_dir):
    with open(text_map, "r") as f:
        for line in f:
            uttid, txt_path = line.strip().split(" ", maxsplit=1)

            with open(txt_path, "r") as fp:
                text = fp.read()

            txt_dir = os.path.join(output_dir, uttid)
            mkdir_if_not_exist(txt_dir)
            with open(os.path.join(txt_dir, "lm_train.txt"), "w") as ofh:
                print(f"{uttid} {text}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Generate text files for each utterance that will be aligned.')
    parser.add_argument('--text_map', type=Path, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the text file will be stored')
    args = parser.parse_args()

    generate_text(args.text_map, args.output_dir)


if __name__ == "__main__":
    main()
