import argparse
import os
from pathlib import Path
import math
import random


def trn_dev_test_split(data_dir, train_dir, dev_dir, test_dir, ratio):
    """
    Split the train/development/test data in data_dir.
    """
    trn_ratio, dev_ratio, test_ratio = [float(r) for r in ratio.split()]
    assert math.isclose(trn_ratio + dev_ratio + test_ratio,
                        1), f"ERROR: ratios don't sum to 1"
    wav_scp = os.path.join(data_dir, "wav.scp")
    text = os.path.join(data_dir, "text")
    utt2spk = os.path.join(data_dir, "utt2spk")

    with open(wav_scp, "r") as f:
        data = f.read().splitlines()
    N = len(data)
    trn_N = int(N * trn_ratio)
    dev_N = int(N * dev_ratio)
    random.shuffle(data)
    dsets = {}
    dsets["train"] = data[:trn_N]
    dsets["dev"] = data[trn_N: trn_N + dev_N]
    dsets["test"] = data[trn_N + dev_N:]

    for key, dset in dsets.items():
        if key == "train":
            _dir = train_dir
        elif key == "dev":
            _dir = dev_dir
        else:
            _dir = test_dir
        dset_wav_scp = os.path.join(_dir, "wav.scp.unsorted")
        dset_text = os.path.join(_dir, "text.unsorted")
        dset_utt2spk = os.path.join(_dir, "utt2spk.unsorted")
        segids = set()
        with open(dset_wav_scp, "w") as f:
            for d in dset:
                segids.add(d.split[0])
                print(d, file=f)
        # Copy corresponding entries based on the selected wav.scp
        with open(dset_text, "w") as ofh:
            with open(text, "r") as f:
                for line in f:
                    line = line.strip()
                    segid = line.split()[0]
                    if segid in segids:
                        print(line, file=ofh)
        with open(dset_utt2spk, "w") as ofh:
            with open(utt2spk, "r") as f:
                for line in f:
                    line = line.strip()
                    segid = line.split()[0]
                    if segid in segids:
                        print(line, file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Train the biased ngram language models using kenlm')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='The data directory in which the raw, train, dev and text data are stored.')
    parser.add_argument('--train_dir', type=Path, required=True,
                        help='The destination directory for storing the training data\'s wav.scp, text, etc.')
    parser.add_argument('--dev_dir', type=Path, required=True,
                        help='The destination directory for storing the development data\'s wav.scp, text, etc.')
    parser.add_argument('--test_dir', type=Path, required=True,
                        help='The destination directory for storing the testing data\'s wav.scp, text, etc.')
    parser.add_argument('--ratio', type=str, default="0.7, 0.2, 0.1",
                        help='The ratio of the train/dev/test split, by default is "0.7, 0.2, 0.1".')
    args = parser.parse_args()

    trn_dev_test_split(args.data_dir, args.train_dir,
                       args.dev_dir, args.test_dir, args.ratio)


if __name__ == "__main__":
    main()
