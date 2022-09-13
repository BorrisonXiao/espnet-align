import argparse
import os
from pathlib import Path


def generate_scp(input_dir, output):
    """
    Generate the key file for biased ngram language model training in the format
    "<scpid> <path_to_lm_train>"
    """
    with open(output, "w") as f:
        for scpid in os.listdir(input_dir):
            print(scpid, os.path.join(input_dir, scpid, "lm_train.txt"), file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the keys for biased ngram training')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='Input: The directory containing the script files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output: The generated ngram_keys.scp file.')
    args = parser.parse_args()

    generate_scp(args.input_dir, args.output)


if __name__ == "__main__":
    main()
