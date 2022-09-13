import argparse
import os
from pathlib import Path
import subprocess
from utils import mkdir_if_not_exist


def ngram_train(keyfile, ngram_exp, ngram_num):
    """
    Perform ngram training using kenlm.
    Note that the installation script for kenlm seems to have some bug, hence manual
    installation is recommended.
    """
    with open(keyfile, "r") as f:
        for line in f:
            scpid, scp_fp = line.strip().split(" ", maxsplit=1)
            ngram_dir = os.path.join(ngram_exp, scpid)
            mkdir_if_not_exist(ngram_dir)
            try:
                subprocess.call(
                    f"./local/ngram_train.sh --lm_train_text {scp_fp} --ngram_dir {ngram_dir} --ngram_num {ngram_num}", shell=True)
            except subprocess.CalledProcessError as e:
                print(e.output)
                exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train the biased ngram language models using kenlm')
    parser.add_argument('--keyfile', type=Path, required=True,
                        help='The keyfile generated for training')
    parser.add_argument('--ngram_exp', type=Path, required=True,
                        help='The destination directory for storing the ngram models.')
    parser.add_argument('--ngram_num', type=int, required=True,
                        help='The n in ngram.')
    args = parser.parse_args()

    ngram_train(args.keyfile, args.ngram_exp, args.ngram_num)


if __name__ == "__main__":
    main()
