#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import mkdir_if_not_exist


def parse_stm(file):
    """
    Helper function for parsing a stm file.
    Input is a path to the stm file.
    Returns a sorted list of tuples (sentid, uttid, start, end, text).
    Note that it might run into returning [] due to empty stm file caused by previous execution error,
    e.g. Grid failure.
    """
    res = []
    with open(file, "r") as f:
        for line in f:
            sentid, uttid, start, end, text = line.strip().split(maxsplit=4)
            start, end = float(start), float(end)
            res.append((sentid, uttid, start, end, text))
    return res


def resolve_overlap(prev, next_stm):
    """
    Helper function for resolving the (time) overlap between two stm parses.
    Returns a list of resolved sentences.
    """
    res = []
    found_anchor = False
    overlap_prev_sentidx = 0
    cn = 0

    # Heuristically search for the first 5 sentences at the start of the next stm
    for cn in range(min(5, len(next_stm))):
        if found_anchor:
            break

        _, _, ns, ne, nt = next_stm[cn]
        try:
            prev[-1][3]
        except IndexError as e:
            print(prev)
            raise
        if cn == 0 and prev[-1][3] < ns:
            # No overlap if the start of the next stm is later than the end of the previous one
            for item in prev:
                res.append(item)
            start_sentid = len(res)
            for i, item in enumerate(next_stm):
                _, uttid, start, end, text = item
                it = (f"sent{start_sentid + i:03}", uttid, start, end, text)
                res.append(it)
            return res
        
        # No point keep going forward/backward if the end of the previous stm is earlier than the start of the current next
        if prev[-1][3] < ns:
            break

        # Iterate from the back to find matching start point of the start of the next stm
        for i in reversed(range(len(prev))):
            item = prev[i]
            sentid, uttid, ps, pe, pt = item
            # The current item (in next_stm) does not have a corresponding segment in prev
            if pe < ns:
                if cn == 0:
                    # Store the overlap starting region for later use, i.e. if no textual overlap is found,
                    # sentences in the next_stm will overwrite the prev
                    overlap_prev_sentidx = i + 1
                break
            # If the starting time of two identical text is not too far away
            if pt == nt and abs(pe - ns) > min(pe - ps, ne - ns) / 3:
                for it in prev[:i]:
                    res.append(it)

                overlap_item = (sentid, uttid, ps, ne, pt)
                res.append(overlap_item)
                found_anchor = True
                break
        
    if not found_anchor:
        for it in prev[:overlap_prev_sentidx]:
            # Results in next will overwrite prev
            res.append(it)

    start_sentid = len(res)
    # The out-of-scope usage of variable here is dangerous in programming, but whatever
    for i, item in enumerate(next_stm[cn:]):
        _, uttid, start, end, text = item
        it = (f"sent{start_sentid + i:03}", uttid, start, end, text)
        res.append(it)

    return res


def merge_alignments(input_dir):
    raw_dir = os.path.join(input_dir, "raw")
    output_dir = os.path.join(input_dir, "data")
    mkdir_if_not_exist(output_dir)

    for uttid in os.listdir(raw_dir):
        utt_dir = os.path.join(raw_dir, uttid)

        # In theory the stm files should be sorted already, but just in case
        segs = sorted(os.listdir(utt_dir))
        segs = [(i, os.path.join(utt_dir, s)) for i, s in enumerate(segs)]
        for (i, seg) in segs:
            if i == 0:
                prev = parse_stm(seg)
            else:
                next_stm = parse_stm(seg)
                if next_stm == []:
                    continue
                if prev == []:
                    prev = next_stm
                    continue
                overlap = resolve_overlap(prev, next_stm)
                prev = overlap

        with open(os.path.join(output_dir, uttid + ".stm"), "w") as ofh:
            for item in prev:
                print(" ".join(map(str, item)), file=ofh)


def main():
    """
    Merge the flexible alignment (sentence) results and form one stm file for each utt.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Merge the flexible alignment (sentence) results and form one stm file for each utt.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    args = parser.parse_args()
    merge_alignments(input_dir=args.input_dir)


if __name__ == "__main__":
    main()
