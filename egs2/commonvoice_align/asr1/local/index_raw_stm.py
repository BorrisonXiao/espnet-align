#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import extract_mid, read_raw_stm, mkdir_if_not_exist
import re
from tqdm import tqdm


def _index(refs, hyp, start_pos):
    # Find the index of the sentence in the reference that matches the hypothesis
    # ref: list of sentences
    # hyp: a sentence
    # start_pos: the starting index of the search
    # return: the index of the sentence in the reference that matches the hypothesis
    for i in range(start_pos, len(refs)):
        if refs[i] == hyp:
            return i
    
    # Backoff to the first sentence in the reference
    for i in range(start_pos):
        if refs[i] == hyp:
            return i
    
    raise ValueError(f"Cannot find {hyp} in the reference.")


def match_sents(input_dir, keyfile, output_dir, mid_text_dir):
    # Parse the text_map file
    mid2fp = {}
    for file in os.listdir(mid_text_dir):
        mid = Path(file).stem
        mid2fp[mid] = os.path.join(mid_text_dir, file)

    idx_dir = output_dir / 'idx'
    mkdir_if_not_exist(idx_dir)
    stm_dir = input_dir / 'outputs' / 'data'
    assert os.path.exists(stm_dir), f"stm_dir {stm_dir} does not exist."

    # Map the text in the stm file with its corresponding index (line number) in the meeting level scp.
    logging.info(f"Creating the rstm.idx files in {idx_dir}...")
    for stm_file in tqdm(os.listdir(stm_dir)):
        segid = Path(stm_file).stem
        mid = extract_mid(segid)
        refs = []
        with open(mid2fp[mid], 'r') as f:
            for line in f:
                _, sent = line.strip().split(maxsplit=1)
                sent = re.sub(r'\s+', ' ', sent)
                refs.append(sent)
        
        hyps = read_raw_stm(os.path.join(stm_dir, stm_file))
        start_pos = 0
        with open(idx_dir / f'{segid}.rstm.idx', 'w') as f:
            for i, hyp in enumerate(hyps):
                try:
                    ref_idx = _index(refs, hyp[-1], start_pos)
                    print(ref_idx, file=f)
                    start_pos = ref_idx + 1
                except ValueError:
                    print(os.path.join(stm_dir, stm_file))
                    raise


def main():
    """
    Match the text in the stm file with its corresponding index in the meeting level scp.
    """
    logging.basicConfig(
        level="INFO", format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Match the text in the stm file with its corresponding index in the meeting level scp.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to the input directory with token and alignments.')
    parser.add_argument('--keyfile', type=Path, default=None,
                        help='Optional keyfile for parallelization.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--mid_text_dir', type=Path, required=True,
                        help='The mid_text_dir storing the sentence-splitted text files')
    args = parser.parse_args()
    match_sents(input_dir=args.input_dir, keyfile=args.keyfile,
                 output_dir=args.output_dir, mid_text_dir=args.mid_text_dir)


if __name__ == "__main__":
    main()
