#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Cihan Xiao)
# Apache 2.0

import argparse
from pathlib import Path

DATA_DIR = Path('/home/cxiao7/research/hklegco_icefall/download/hklegco')
SPLITS = ['train', 'test', 'dev-asr', 'dev-mt', 'dev-asr-0',
          'dev-asr-1', 'dev-asr-2', 'dev-mt-0', 'dev-mt-1', 'dev-mt-2']
EPS = 1e-6


def stats(data_dir: Path, punctuated: bool = True):
    asr_dir = data_dir / 'asr'

    total_count = 0
    for split in SPLITS:
        # Step 1: Read the text file
        split_dir = asr_dir / split
        with open(split_dir / 'text', 'r') as f:
            asr_text = f.readlines()
        asr_text = [line.strip() for line in asr_text]
        try:
            asr_text = [line.split(' ', maxsplit=1)[1] for line in asr_text]
        except IndexError:
            breakpoint()

        # Step 2: Count the number of tokens
        count = 0
        for line in asr_text:
            if punctuated:
                count += len(line)
            else:
                raise NotImplementedError
        print(f'{split} has {count//1000}k Cantonese tokens')

        if split in ['train', 'dev-asr', 'dev-mt', "test"]:
            total_count += count
    print(f'Total: {total_count//1000}k Cantonese tokens')

    # Count the English tokens from the stm file
    print('')
    st_dir = data_dir / 'st'
    total_en_count = 0
    for split in SPLITS:
        with open(st_dir / f"st-can2eng.{split}.stm", 'r') as f:
            stm_text = f.readlines()
        count = sum([len(line.split()[5:]) for line in stm_text])
        print(f'{split} has {count//1000}k English tokens')
        if split in ['train', 'dev-asr', 'dev-mt', "test"]:
            total_en_count += count
    print(f'Total: {total_en_count//1000}k English tokens')


def main():
    parser = argparse.ArgumentParser(
        description='Get token counts of the splits.')
    parser.add_argument('--data_dir', type=Path, default=DATA_DIR,
                        help='The splitted data directory.')
    parser.add_argument('--punc', action='store_false',
                        help='Whether the text is punctuated.')
    args = parser.parse_args()

    stats(data_dir=args.data_dir, punctuated=args.punc)


if __name__ == "__main__":
    main()
