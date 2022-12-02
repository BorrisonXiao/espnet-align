#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import segid2uttid, mkdir_if_not_exist


def clean_salign(input_dir, output_dir, text_map, segments):
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
            segid, hyp = line.strip().split(maxsplit=1)
            hyps[segid] = hyp.split()
    with open(alignments_fp, "r") as f:
        alms = {}
        for line in f:
            segid, alignment = line.strip().split(maxsplit=1)
            alms[segid] = alignment.split()
    assert [len(hyps[k]) == len(alms[k]) - 1 for k in hyps]

    for segid, hyp in hyps.items():
        # Each segment in the hypothesis is shifted by their starting time in the utterance
        shift = float(segs[segid][0])
        uttid = segid2uttid(segid)

        _dir = os.path.join(output_dir, uttid)
        mkdir_if_not_exist(_dir)
        stm = os.path.join(_dir, f"{segid}.stm")
        with open(stm, "w") as ofh:
            ref = []
            with open(refs[uttid], "r") as f:
                for line in f:
                    _, sent = line.strip().split(maxsplit=1)
                    ref.append(sent)

            # Brute-force search, could be optimized
            matched_seg_idx = 0
            matched_hyp_idx = 0
            breaks = []
            # Sort by length to prioritize longer mappings
            ref.sort(key=len, reverse=True)

            while matched_hyp_idx < len(hyp) - 1:
                matched = False
                for sent_id, sent in enumerate(ref):
                    sent = sent.strip().split()
                    sent_len = len(sent)
                    start = matched_hyp_idx
                    end = sent_len + matched_hyp_idx
                    hyp_seg = hyp[start:end]
                    # Replace low-freq words with unks in the refernce for matching
                    for i, c in enumerate(hyp_seg):
                        if c == '<unk>':
                            sent[i] = '<unk>'

                    if hyp_seg == sent:
                        matched = True
                        matched_hyp_idx = end
                        matched_seg_idx += 1
                        ref.pop(sent_id)
                        # Record the starting index of a sentence
                        breaks.append((start, sent))
                assert matched, f"Cannot find match for {hyp[start:]}"

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
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Clean up flexible alignment results and build sentence alignments.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--text_map', type=Path, required=True,
                        help='The text_map file storing pointers to the sentence-splitted text files')
    parser.add_argument('--segments', type=Path, required=True,
                        help='The segments file storing the start and end time of each segment in the utterance.')
    args = parser.parse_args()
    clean_salign(input_dir=args.input_dir,
                 output_dir=args.output_dir, text_map=args.text_map, segments=args.segments)


if __name__ == "__main__":
    main()
