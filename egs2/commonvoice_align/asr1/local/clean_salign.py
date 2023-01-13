#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import segid2uttid, mkdir_if_not_exist


def dp_sent(hyp, ref):
    # Dynamic programming algorithm for breaking the hypothesis into sentences in the reference
    # The idea is based on the CKY algorithm for parsing
    # hyp: list of chars
    # ref: list of sentences, can be construed as a list of rules in the CKY algorithm
    # return: list of (start, end) indices of sentences in the hypothesis

    # Initialize the DP table
    # dp[i][j] is the list of (start, end) indices of sentences in the reference that can be
    # matched to the hypothesis from i to j
    dp = [[[] for _ in range(len(hyp) + 1)] for _ in range(len(hyp) + 1)]

    # Base case: dp[i][i] = [[]] for all i (not sure if this is necessary)
    for i in range(len(hyp) + 1):
        dp[i][i] = [[]]

    # Fill the DP table
    # Unlike the CKY algorithm, we don't need to increment the step by 1 every time
    # instead we can increment by the length of the possible sentences in the reference
    start_idx = [(0, (0, 0), 0)]
    back_idx = None
    while start_idx != []:
        start, prev_pos, ref_start_idx = start_idx.pop(0)
        for end in range(start + 1, len(hyp) + 1):
            for ref_idx, sent in enumerate(ref):
                # Constraint the search space to be sentences in the reference after the previous match
                if ref_idx < ref_start_idx:
                    continue
                sent = sent.strip().split()
                sent_len = len(sent)
                if end - start < sent_len:
                    continue
                if hyp[start:end] == sent:
                    if (start, end, prev_pos) in dp[start][end]:
                        continue
                    dp[start][end].append((start, end, prev_pos))
                    if end == len(hyp):
                        back_idx = (start, end)
                    if (end, (start, end)) not in start_idx:
                        start_idx.append((end, (start, end), ref_idx + 1))

    # Backtrack to find the best match
    # The best match is the one that has the most sentences in the reference
    # If there are multiple matches with the same number of sentences, we choose the one
    # that has the longest sentences
    res = []
    prev = back_idx
    while prev != (0, 0) and prev is not None:
        start, end = prev
        res.append((start, hyp[start:end]))
        # Selecting the first one as it's not very likely to have multiple parses
        prev = dp[start][end][0][2]

    return list(reversed(res))


def clean_salign(input_dir, output_dir, text_map, segments, keyfile):
    if keyfile is not None:
        with open(keyfile, "r") as f:
            keys = set([line.strip() for line in f])
    else:
        keys = None

    output_dir = os.path.join(output_dir, "raw")
    alignments_fp = os.path.join(input_dir, "alignments")
    token_fp = os.path.join(input_dir, "token")
    with open(text_map, "r") as f:
        refs = {}
        for line in f:
            uttid, ref_fp = line.strip().split(maxsplit=1)
            refs[uttid] = ref_fp
    with open(segments, "r") as f:
        segs = {}
        for line in f:
            segid, uttid, start, end = line.strip().split()
            segs[segid] = (start, end)
    with open(token_fp, "r") as f:
        hyps = {}
        for line in f:
            try:
                segid, hyp = line.strip().split(maxsplit=1)
            except ValueError:
                logging.warning(f"ValueError: Empty output for {line.strip()}...")
                continue
            hyps[segid] = hyp.split()
    with open(alignments_fp, "r") as f:
        alms = {}
        for line in f:
            try:
                segid, alignment = line.strip().split(maxsplit=1)
            except ValueError:
                continue
            alms[segid] = alignment.split()
    assert [len(hyps[k]) == len(alms[k]) - 1 for k in hyps]

    for segid, hyp in hyps.items():
        # Each segment in the hypothesis is shifted by their starting time in the utterance
        shift = float(segs[segid][0])
        uttid = segid2uttid(segid)

        if keys is not None and uttid not in keys:
            continue

        logging.info(f"Processing {segid}...")

        _dir = os.path.join(output_dir, uttid)
        mkdir_if_not_exist(_dir)
        stm = os.path.join(_dir, f"{segid}.stm")
        with open(stm, "w") as ofh:
            ref = []
            with open(refs[uttid], "r") as f:
                for line in f:
                    _, sent = line.strip().split(maxsplit=1)
                    ref.append(sent)

            # Use a CKY-like algorithm to break the hypothesis into sentences in the reference
            breaks = dp_sent(hyp, ref)

            sent_alms = [(alms[segid][i], " ".join(s)) for (i, s) in breaks]

            for i in range(len(sent_alms)):
                begin = float(sent_alms[i][0])
                if i < len(sent_alms) - 1:
                    end = float(sent_alms[i + 1][0])
                else:
                    # Note that the last element in alms[segid] is the length of that audio
                    end = float(alms[segid][-1])
                print(
                    f"sent{i:03} {uttid} {begin + shift:.2f} {end + shift:.2f} {sent_alms[i][1]}", file=ofh)


def main():
    """
    Clean up flexible alignment results and build sentence alignments.
    """
    logging.basicConfig(
        level="INFO", format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Clean up flexible alignment results and build sentence alignments.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    parser.add_argument('--keyfile', type=Path, default=None,
                        help='Optional keyfile for parallelization.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--text_map', type=Path, required=True,
                        help='The text_map file storing pointers to the sentence-splitted text files')
    parser.add_argument('--segments', type=Path, required=True,
                        help='The segments file storing the start and end time of each segment in the utterance.')
    args = parser.parse_args()
    clean_salign(input_dir=args.input_dir, keyfile=args.keyfile,
                 output_dir=args.output_dir, text_map=args.text_map, segments=args.segments)


if __name__ == "__main__":
    main()
