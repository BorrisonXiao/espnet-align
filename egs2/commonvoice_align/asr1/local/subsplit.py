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
from utils import read_seg, read_annot, mkdir_if_not_exist, cut_uttid_len
from matplotlib import pyplot as plt
import numpy as np
import shutil

SPLITS = ['train', 'test', 'other', 'dev-asr', 'dev-mt', 'dev-other']
GENDER = ['Female', 'Male']
EPS = 1e-6
RATIO = [0.3, 0.3, 0.4]


def _divide(spkfile: Path, data_dir: Path, output_dir: Path, punctuated: bool = False):
    subsplit(data_dir, output_dir, 'dev-asr')
    subsplit(data_dir, output_dir, 'dev-mt')


def subsplit(data_dir: Path, output_dir: Path, dset: str):
    data_entry = defaultdict(dict)
    # Step 1: Read the data directory
    asr_dir = data_dir / 'asr'
    segments_fp = asr_dir / dset / 'segments'
    text_fp = asr_dir / dset / 'text'
    utt2spk_fp = asr_dir / dset / 'utt2spk'

    # Step 1: Read the files simultaneously to form a data entry
    with open(segments_fp, 'r') as seg_f, open(text_fp, 'r') as text_f, open(utt2spk_fp, 'r') as utt2spk_f:
        for seg_line, text_line, utt2spk_line in zip(seg_f, text_f, utt2spk_f):
            segid, uttid, start, end = seg_line.strip().split()
            text = text_line.strip().split(maxsplit=1)[1]
            spkid = utt2spk_line.strip().split()[1]
            entry_id = (uttid, start, end)
            data_entry[entry_id]["segment"] = seg_line.strip()
            data_entry[entry_id]["text"] = text_line.strip()
            data_entry[entry_id]["spkid"] = utt2spk_line.strip()

    # Step 2: Read the stm file and add the corresponding data entry
    asr_stm_fp = data_dir / 'st' / f"asr-can.{dset}.stm"
    st_stm_fp = data_dir / 'st' / f"st-can2eng.{dset}.stm"
    with open(asr_stm_fp, 'r') as asr_stm_f, open(st_stm_fp, 'r') as st_stm_f:
        for asr_line, st_line in zip(asr_stm_f, st_stm_f):
            asr_wav_fp, asr_channel, spkid, asr_start, asr_end, asr_text = asr_line.strip().split(maxsplit=5)
            st_wav_fp, st_channel, spkid, st_start, st_end, st_text = st_line.strip().split(maxsplit=5)
            uttid = cut_uttid_len(asr_wav_fp.split('/')[-1].split('.')[0])
            entry_id = (uttid, asr_start, asr_end)
            if entry_id not in data_entry:
                uttid = cut_uttid_len(asr_wav_fp.split('/')[-1].split('.')[0], 64, 64)
                entry_id = (uttid, asr_start, asr_end)
            assert entry_id in data_entry, f"Entry {entry_id} not found in {dset}"
            data_entry[entry_id]["asr_stm"] = asr_line.strip()
            data_entry[entry_id]["st_stm"] = st_line.strip()

    # Step 3: Group the data entry by speaker
    spk2entry = defaultdict(list)
    for entry_id in data_entry.keys():
        spkid = data_entry[entry_id]["spkid"].strip().split()[1]
        spk2entry[spkid].append(entry_id)

    # Step 4: Randomly sample the data entry for each speaker
    spk2subsplit = [[] for _ in range(len(RATIO))]
    for spkid in spk2entry.keys():
        entry_ids = spk2entry[spkid]
        np.random.shuffle(entry_ids)
        # Partition the entries by ratio
        pos = [0]
        for ratio in RATIO[:-1]:
            pos.append(int(len(entry_ids) * ratio) + pos[-1])
        pos.append(len(entry_ids))
        for i in range(len(RATIO)):
            toadd = entry_ids[pos[i]:pos[i + 1]]
            if len(toadd) > 0:
                toadd.sort()
            spk2subsplit[i].extend(toadd)

    # Step 5: Write the data entry to the output directory
    for i, split in enumerate(spk2subsplit):
        out_asr_dir = output_dir / 'asr' / f"{dset}-{i}"
        out_st_dir = output_dir / 'st'

        mkdir_if_not_exist(out_asr_dir)
        mkdir_if_not_exist(out_st_dir)

        out_segments_fp = out_asr_dir / "segments"
        out_text_fp = out_asr_dir / "text"
        out_utt2spk_fp = out_asr_dir / "utt2spk"
        out_asr_stm_fp = out_st_dir / f"asr-can.{dset}-{i}.stm"
        out_st_stm_fp = out_st_dir / f"st-can2eng.{dset}-{i}.stm"

        tot_time = 0
        with open(out_segments_fp, 'w') as seg_f, open(out_text_fp, 'w') as text_f, open(out_utt2spk_fp, 'w') as utt2spk_f, \
                open(out_asr_stm_fp, 'w') as asr_stm_f, open(out_st_stm_fp, 'w') as st_stm_f:
            for entry_id in split:
                # Also accumulate the time for each set
                tot_time += float(data_entry[entry_id]["segment"].strip().split()[3]) - float(
                    data_entry[entry_id]["segment"].strip().split()[2])
                print(data_entry[entry_id]["segment"], file=seg_f)
                print(data_entry[entry_id]["text"], file=text_f)
                print(data_entry[entry_id]["spkid"], file=utt2spk_f)
                print(data_entry[entry_id]["asr_stm"], file=asr_stm_f)
                print(data_entry[entry_id]["st_stm"], file=st_stm_f)
        print(f"Total time for {dset}-{i}: {tot_time / 3600:.2f} hours")

    # Step 6: Copy the wav.scp file
    for i in range(len(RATIO)):
        shutil.copy(data_dir / 'asr' / dset / 'wav.scp', output_dir / 'asr' / f"{dset}-{i}")


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
    ax[0][0].set_xlim(0, 55)
    ax[0][1].set_xlim(0, 55)
    ax[1][0].set_xlim(0, 55)
    ax[1][1].set_xlim(0, 55)

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
    txt_len_dist = {dset: [] for dset in SPLITS}
    for dset in SPLITS:
        with open(data_dir / 'st' / f'st-can2eng.{dset}.stm', 'r') as f:
            lines = f.readlines()
        for line in lines:
            txt_len = len(line.split()[5:])
            txt_len_dist[dset].append(txt_len)
        token_count[dset] = sum([len(line.split()[5:]) for line in lines])
        total_token_count += token_count[dset]
    for dset in SPLITS:
        if token_count[dset] == 0:
            continue
        print(f'The number of tokens in {dset} is {token_count[dset]:,} / {token_count[dset] / (total_token_count) * 100:.2f}%.')
        print(f"The average number of tokens per utterance in {dset} is {token_count[dset] / (sent_count[dset]):.2f}.")
    print(f'The total number of tokens is {total_token_count:,}.')
    print(f'The average number of tokens per utterance is {total_token_count / total_sent_count:.2f}.\n')

    # Plot the distribution of the English token-level text length
    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    ax[0][0].hist(txt_len_dist['train'], bins=50, label='Train')
    ax[0][1].hist(txt_len_dist['test'], bins=50, label='Test')
    ax[1][0].hist(txt_len_dist['dev-asr'], bins=50, label='Dev-ASR')
    ax[1][1].hist(txt_len_dist['dev-mt'], bins=50, label='Dev-MT')
    ax[0][0].set_title('English Token-Level Text Length Distribution for Train')
    ax[0][1].set_title('English Token-Level Text Length Distribution for Test')
    ax[1][0].set_title('English Token-Level Text Length Distribution for Dev-ASR')
    ax[1][1].set_title('English Token-Level Text Length Distribution for Dev-MT')
    ax[0][0].set_xlabel('English Token-Level Text Length')
    ax[0][1].set_xlabel('English Token-Level Text Length')
    ax[1][0].set_xlabel('English Token-Level Text Length')
    ax[1][1].set_xlabel('English Token-Level Text Length')
    ax[0][0].set_xlim(0, 80)
    ax[0][1].set_xlim(0, 80)
    ax[1][0].set_xlim(0, 80)
    ax[1][1].set_xlim(0, 80)

    fig.savefig(output_dir / 'txt_len_dist.png')

def main():
    parser = argparse.ArgumentParser(
        description='Divide the dev-asr and the dev-mt set into smaller sets to reduce their size and get some stats after the division.')
    parser.add_argument('--annotfile', type=Path, required=True,
                        help='The full path to the annotation file that stores the gender information.')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='The data directory.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the output directory.')
    parser.add_argument('--punc', action='store_true',
                        help='Whether the text is punctuated.')
    args = parser.parse_args()

    _divide(spkfile=args.annotfile, data_dir=args.data_dir, output_dir=args.output_dir, punctuated=args.punc)


if __name__ == "__main__":
    main()
