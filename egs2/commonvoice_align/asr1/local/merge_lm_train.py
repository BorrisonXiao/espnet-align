import argparse
from pathlib import Path
import os


def merge_lm_train(utts_dir, output, sentence):
    with open(output, "w") as ofh:
        for scpid in os.listdir(utts_dir):
            lm_train = os.path.join(utts_dir, scpid, "lm_train.txt")
            with open(lm_train, "r") as f:
                if sentence:
                    for line in f:
                        print(f"{line.strip()}", file=ofh)
                else:
                    txt = f.read().rstrip()
                    print(f"{txt}", file=ofh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge individual lm_train.txt to create universal token_list.')
    parser.add_argument('--utts_dir',
                        type=Path,
                        required=True,
                        help='The directory containing the input lm_train.txt files.')
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='The output file in which the merged texts are stored.')
    parser.add_argument('--sentence', action='store_true',
                        help='If true, the lm_train.txt will be sentence-level')
    args = parser.parse_args()

    merge_lm_train(args.utts_dir, args.output, args.sentence)
