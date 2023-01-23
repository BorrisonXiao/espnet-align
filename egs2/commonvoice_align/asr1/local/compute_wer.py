#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
from utils import read_aligned_result
from tqdm import tqdm
import logging


def wer(c, s, i, d):
    """
    Compute the word error rate based on CSID.
    Note that the length of the reference is simply c + s + d.
    """
    return (s + i + d) / (c + s + d)


def compute_wer(input, output):
    """
    Compute the WER for each utterance based on the align-text script + wer_per_utt_details.pl's output.
    """
    alignment_results = read_aligned_result(input)
    with open(output, "w") as f:
        for res in tqdm(alignment_results):
            uttid, _, _, _, csid = res
            _wer = wer(*csid)
            print(f"{uttid} {_wer:.4f}", file=f)


def main():
    """
    Compute the WER for each utterance based on the align-text script + wer_per_utt_details.pl's output.
    """
    logging.basicConfig(
        level="INFO", format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description="Compute the WER for each utterance based on the align-text script + wer_per_utt_details.pl's output.")
    parser.add_argument('--input', type=Path, required=True,
                        help='The full path to the input file in which the alignment results are stored.')
    parser.add_argument('--output', type=Path, required=True,
                        help='The output file storing the wer in the format (uttid, wer).')
    args = parser.parse_args()
    compute_wer(input=args.input, output=args.output)


if __name__ == "__main__":
    main()
