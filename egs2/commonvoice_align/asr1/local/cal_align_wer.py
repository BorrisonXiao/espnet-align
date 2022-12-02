#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import mkdir_if_not_exist
import editdistance


def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score.

    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """

    word_eds, word_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)


def align_wer(input_dir, textmap):
    hyps = []
    refs = []
    maps = {}

    with open(textmap, "r") as f:
        for line in f:
            uttid, text_fp = line.strip().split(maxsplit=1)
            maps[uttid] = text_fp

    for aligned in os.listdir(input_dir):
        with open(os.path.join(input_dir, aligned), "r") as f:
            hyps.append([])
            refs.append([])

            for line in f:
                _, uttid, _, _, text = line.strip().split(maxsplit=4)
                hyps[-1] += text.split()
            hyps[-1] = " ".join(hyps[-1])

            with open(maps[uttid], "r") as fp:
                for l in fp:
                    _, t = l.strip().split(maxsplit=1)
                    refs[-1] += t.split()
            refs[-1] = " ".join(refs[-1])
                
    print(calculate_wer(hyps, refs))


def main():
    """
    Merge the flexible alignment (sentence) results and form one stm file for each utt.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Merge the flexible alignment (sentence) results and form one stm file for each utt.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    parser.add_argument('--textmap', type=Path, required=True,
                        help='The full path to the textmap.')
    args = parser.parse_args()
    align_wer(input_dir=args.input_dir, textmap=args.textmap)


if __name__ == "__main__":
    main()