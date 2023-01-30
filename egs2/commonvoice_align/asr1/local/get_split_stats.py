#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Cihan Xiao)
# Apache 2.0

import argparse
from pathlib import Path
from collections import defaultdict, Counter
import re
from tqdm import tqdm
import os
from utils import read_seg, read_annot, mkdir_if_not_exist
from matplotlib import pyplot as plt
import numpy as np

SPLITS = ['train', 'test', 'other', 'dev-asr', 'dev-mt', 'dev-other']
GENDER = ['Female', 'Male']
EPS = 1e-6


def stats(spkfile: Path, data_dir: Path, output_dir: Path, punctuated: bool = False):
    # Step 1: Read the segments file
    asr_dir = data_dir / 'asr'
    segments = {}
    segments_fps = {dset: asr_dir / dset / 'segments' for dset in os.listdir(asr_dir)}
    for dset, segments_fp in segments_fps.items():
        segments[dset] = read_seg(segments_fp)

    # Step 2: Read the annotation file
    spk2info = read_annot(spkfile)
    spk2gender = {spkid: spk2info[spkid]['gender'] for spkid in spk2info.keys()}
    gender_utt_dist = {dset: [0, 0] for dset in SPLITS}
    gender_spk_occ = {dset: [set(), set()] for dset in SPLITS}

    # Step 3: Plot the distribution of gender for each split
    split_duration = {dset: 0 for dset in SPLITS}
    for dset, segment in segments.items():
        for (segid, uttid, start, end) in segment:
            spkid = segid.split('_')[0]
            gender = spk2gender[spkid]
            gender_utt_dist[dset][gender] += 1
            gender_spk_occ[dset][gender].add(spkid)
            split_duration[dset] += float(end) - float(start)
        print(f'{dset} duration: {split_duration[dset] / 3600:.2f} hours')
    gender_spk_dist = {dset: [len(gender_spk_occ[dset][0]), len(gender_spk_occ[dset][1])] for dset in SPLITS}
    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    female_counts = [gender_spk_dist[dset][0] for dset in SPLITS]
    male_counts = [gender_spk_dist[dset][1] for dset in SPLITS]
    X = np.arange(len(SPLITS))
    ax[0][0].bar(X, female_counts, label='Female', width=0.4)
    ax[0][0].bar(X + 0.4, male_counts, label='Male', width=0.4)
    for i, v in enumerate(female_counts):
        if v == 0:
            continue
        ax[0][0].text(i - 0.15, v + 1, str(v))
    for i, v in enumerate(male_counts):
        if v == 0:
            continue
        ax[0][0].text(i + 0.25, v + 1, str(v))
    ax[0][0].set_xticks(X + 0.4 / 2)
    ax[0][0].set_xticklabels(SPLITS, rotation=45)
    ax[0][0].set_title('Gender Counts')
    ax[0][0].legend()

    female_utt_counts = [gender_utt_dist[dset][0] for dset in SPLITS]
    male_utt_counts = [gender_utt_dist[dset][1] for dset in SPLITS]
    X = np.arange(len(SPLITS))
    ax[0][1].bar(X, female_utt_counts, label='Female', width=0.4)
    ax[0][1].bar(X + 0.4, male_utt_counts, label='Male', width=0.4)
    for i, v in enumerate(female_utt_counts):
        if v == 0:
            continue
        ax[0][1].text(i - 0.3, v + 500, str(v))
    for i, v in enumerate(male_utt_counts):
        if v == 0:
            continue
        ax[0][1].text(i + 0.2, v + 500, str(v))
    ax[0][1].set_xticks(X + 0.4 / 2)
    ax[0][1].set_xticklabels(SPLITS, rotation=45)
    ax[0][1].set_title('Gender-based Utterance Counts')
    ax[0][1].legend()

    # Plot the normalized distribution
    sum_counts = np.array([female_counts[i] + male_counts[i] for i in range(len(SPLITS))])
    ax[1][0].bar(X, np.array(female_counts) / sum_counts, label='Female', width=0.4)
    ax[1][0].bar(X + 0.4, np.array(male_counts) / sum_counts, label='Male', width=0.4)
    for i, v in enumerate(female_counts):
        if v == 0:
            continue
        ax[1][0].text(i - 0.15, v + 1, str(v))
    for i, v in enumerate(male_counts):
        if v == 0:
            continue
        ax[1][0].text(i + 0.25, v + 1, str(v))
    ax[1][0].set_xticks(X + 0.4 / 2)
    ax[1][0].set_xticklabels(SPLITS, rotation=45)
    ax[1][0].set_title('Gender Distribution')
    ax[1][0].legend()

    sum_utt_counts = np.array([female_utt_counts[i] + male_utt_counts[i] for i in range(len(SPLITS))])
    ax[1][1].bar(X, np.array(female_utt_counts) / sum_utt_counts, label='Female', width=0.4)
    ax[1][1].bar(X + 0.4, np.array(male_utt_counts) / sum_utt_counts, label='Male', width=0.4)
    for i, v in enumerate(female_counts):
        if v == 0:
            continue
        ax[1][1].text(i - 0.15, v + 1, str(v))
    for i, v in enumerate(male_counts):
        if v == 0:
            continue
        ax[1][1].text(i + 0.25, v + 1, str(v))
    ax[1][1].set_xticks(X + 0.4 / 2)
    ax[1][1].set_xticklabels(SPLITS, rotation=45)
    ax[1][1].set_title('Gender-based Utterance Distribution')
    ax[1][1].legend()

    fig.savefig(output_dir / 'spk_dist.png')

    # Step 4: Plot the distribution of the utterance length
    utt_len_dist = {dset: [] for dset in SPLITS}
    for dset, segment in segments.items():
        for (segid, uttid, start, end) in segment:
            utt_len = end - start
            if utt_len > 60:
                print(segid)
            utt_len_dist[dset].append(utt_len)
    
    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    ax[0][0].hist(utt_len_dist['train'], bins=50, label='Train')
    ax[0][1].hist(utt_len_dist['test'], bins=50, label='Test')
    ax[1][0].hist(utt_len_dist['dev-asr'], bins=50, label='Dev-ASR')
    ax[1][1].hist(utt_len_dist['dev-mt'], bins=50, label='Dev-MT')
    ax[0][0].set_title('Utterance Length Distribution for Train')
    ax[0][1].set_title('Utterance Length Distribution for Test')
    ax[1][0].set_title('Utterance Length Distribution for Dev-ASR')
    ax[1][1].set_title('Utterance Length Distribution for Dev-MT')
    ax[0][0].set_xlabel('Utterance Length (seconds)')
    ax[0][1].set_xlabel('Utterance Length (seconds)')
    ax[1][0].set_xlabel('Utterance Length (seconds)')
    ax[1][1].set_xlabel('Utterance Length (seconds)')
    ax[0][0].set_xlim(0, 80)
    ax[0][1].set_xlim(0, 80)
    ax[1][0].set_xlim(0, 80)
    ax[1][1].set_xlim(0, 80)

    fig.savefig(output_dir / 'utt_len_dist.png')

    # Step 5: Get sentence-level and bilingual token-level text information
    # Step 5.1: Get the sentence-level text information
    total_sent_count = 0
    sent_count = Counter()
    token_count = Counter()
    total_token_count = 0
    for dset in SPLITS:
        with open(data_dir / 'st' / f'asr-can.{dset}.stm', 'r') as f:
            lines = f.readlines()
        sent_count[dset] = len(lines)
        total_sent_count += sent_count[dset]
        # print(dset)
        # print(sum([len(line.split()[-1]) for line in lines]))
        token_count[dset] = sum([len(line.split()[5:]) for line in lines]) if not punctuated else sum([len(line.split()[-1]) for line in lines])
        total_token_count += token_count[dset]
    for dset in SPLITS:
        print(f'The number of sentences in {dset} is {sent_count[dset]:,} / {sent_count[dset] / total_sent_count * 100:.2f}%.')
    print(f'The total number of sentences is {total_sent_count:,}.\n')
    for dset in SPLITS:
        if token_count[dset] == 0:
            continue
        print(f'The number of tokens in {dset} is {token_count[dset]:,} / {token_count[dset] / (total_token_count) * 100:.2f}%.')
        print(f"The average number of tokens per utterance in {dset} is {token_count[dset] / (sent_count[dset]):.2f}.")
    print(f'The total number of tokens is {total_token_count:,}.')
    print(f'The average number of tokens per utterance is {total_token_count / total_sent_count:.2f}.\n')

    token_count = Counter()
    total_token_count = 0
    for dset in SPLITS:
        with open(data_dir / 'st' / f'st-can2eng.{dset}.stm', 'r') as f:
            lines = f.readlines()
        token_count[dset] = sum([len(line.split()[5:]) for line in lines])
        total_token_count += token_count[dset]
    for dset in SPLITS:
        if token_count[dset] == 0:
            continue
        print(f'The number of tokens in {dset} is {token_count[dset]:,} / {token_count[dset] / (total_token_count) * 100:.2f}%.')
        print(f"The average number of tokens per utterance in {dset} is {token_count[dset] / (sent_count[dset]):.2f}.")
    print(f'The total number of tokens is {total_token_count:,}.')
    print(f'The average number of tokens per utterance is {total_token_count / total_sent_count:.2f}.\n')


def main():
    parser = argparse.ArgumentParser(
        description='Get some statistics of the splits.')
    parser.add_argument('--annotfile', type=Path, required=True,
                        help='The full path to the annotation file that stores the gender information.')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='The splitted data directory.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the output directory.')
    parser.add_argument('--punc', action='store_true',
                        help='Whether the text is punctuated.')
    args = parser.parse_args()

    stats(spkfile=args.annotfile, data_dir=args.data_dir, output_dir=args.output_dir, punctuated=args.punc)


if __name__ == "__main__":
    main()
