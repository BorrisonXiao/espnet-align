#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
from tqdm import tqdm
from pathlib import Path
from utils import read_seg, mkdir_if_not_exist, read_text, read_utt2spk, read_aligned_result
import os
import random
from collections import defaultdict, Counter
import numpy as np

random.seed(10)


def valid(wer, wer_threshold, info, ratio, duration):
    ref, hyp, op, csid = info
    c, s, i, d = csid
    ref_len = c + s + d
    if hyp == "":
        return False
    if wer > wer_threshold:
        return False
    if c <= 2:
        return False
    if ref_len < 8 and c <= 3:
        return False
    if wer >= 0.85:
        if ref_len >= 20 and c <= 8:
            return False
    # Find the maximum number of consecutive insertions and deletions
    max_i, max_d = 0, 0
    cur_i, cur_d = 0, 0
    for o in op:
        if o == "I":
            cur_i += 1
            cur_d = 0
        elif o == "D":
            cur_d += 1
            cur_i = 0
        else:
            cur_i, cur_d = 0, 0
        max_i = max(max_i, cur_i)
        max_d = max(max_d, cur_d)
    if max_i >= 8 or max_d >= 8:
        return False
    # After discussion it was decided to filter out too long utterances
    if duration > 50:
        return False
    return True


def filter(input_dir, output_dir, wer, aligned_file, threshold, dumpdir):
    asr_dir = input_dir / "asr"
    stm_dir = input_dir / "st"
    segid2stmidx = input_dir / "metadata" / "segid2stmidx"

    # Read the segid2stmidx file
    with open(segid2stmidx, 'r') as f:
        lines = f.readlines()
    segid2stmidx = {}
    for line in lines:
        line = line.strip()
        segid, stmidx = line.split()
        segid2stmidx[segid] = int(stmidx)

    # Read the aligned file
    aligned = read_aligned_result(aligned_file)
    segid2aligned = {}
    for elem in aligned:
        segid, ref, hyp, op, csid = elem
        segid2aligned[segid] = (ref, hyp, op, csid)

    # Read the wer file
    with open(wer, 'r') as f:
        lines = f.readlines()
    segid2wer = {}
    for line in lines:
        line = line.strip()
        segid, wer = line.split()
        segid2wer[segid] = float(wer)

    out_asr_dir = output_dir / "asr"
    out_stm_dir = output_dir / "st"
    out_segid2stmidx = output_dir / "metadata" / "segid2stmidx"
    mkdir_if_not_exist(out_asr_dir)
    mkdir_if_not_exist(out_stm_dir)
    mkdir_if_not_exist(out_segid2stmidx.parent)

    # Used to filter the wav.scp file
    filtered_uttids = set()
    # Used to filter the stm file
    filtered_stmidxs = set()

    # Filter the segments, text and utt2spk file
    segs = read_seg(asr_dir / "segments")
    total_duration = 0
    token_count = 0
    valid_segids = set()
    if dumpdir:
        segid2time = {}
    for seg in tqdm(segs):
        segid, uttid, start, end = seg
        csid = segid2aligned[segid][-1]
        c, s, _, d = csid
        ref_len = c + s + d
        total_duration += (end - start)
        token_count += ref_len
    # Use the average duration to text ratio to filter the segments as well
    avg_dur2text_ratio = token_count / total_duration
    with open(out_asr_dir / "segments", 'w') as seg_f:
        for seg in tqdm(segs):
            segid, uttid, start, end = seg
            start, end = str(start), str(end)
            if dumpdir:
                segid2time[segid] = (start, end)
            wer = segid2wer[segid]
            if valid(wer=wer, wer_threshold=threshold, info=segid2aligned[segid], ratio=avg_dur2text_ratio, duration=float(end)-float(start)):
                filtered_uttids.add(uttid)
                filtered_stmidxs.add(segid2stmidx[segid])
                valid_segids.add(segid)
                print(f"{segid} {uttid} {start} {end}", file=seg_f)

    texts = read_text(asr_dir / "text")
    if dumpdir:
        segid2ref = {}
    with open(out_asr_dir / "text", "w") as text_f:
        for text in tqdm(texts):
            segid, txt = text
            if dumpdir:
                segid2ref[segid] = txt
            wer = segid2wer[segid]
            if segid in valid_segids:
                print(f"{segid} {txt}", file=text_f)

    utt2spks = read_utt2spk(asr_dir / "utt2spk", ignore_seg=False, return_list=True)
    if dumpdir:
        spk_full_wer = defaultdict(list)
        spk_valid_wer = defaultdict(list)
        spk_duration = {}
        spk_uttnum = Counter()
    with open(out_asr_dir / "utt2spk", "w") as utt2spk_f:
        for elem in tqdm(utt2spks):
            segid, spkid = elem
            if dumpdir:
                spk_full_wer[spkid].append(segid2wer[segid])
            wer = segid2wer[segid]
            if segid in valid_segids:
                print(f"{segid} {spkid}", file=utt2spk_f)
                if dumpdir:
                    spk_duration[spkid] = spk_duration.get(
                        spkid, 0) + float(segid2time[segid][1]) - float(segid2time[segid][0])
                    spk_uttnum[spkid] += 1
                    spk_valid_wer[spkid].append(segid2wer[segid])

    # Filter the wav.scp file
    if dumpdir:
        uttid2wavfp = {}
    with open(asr_dir / "wav.scp", 'r') as wav_f:
        with open(out_asr_dir / "wav.scp", 'w') as out_wav_f:
            for line in tqdm(wav_f):
                uttid, wavfp = line.strip().split()
                if dumpdir:
                    uttid2wavfp[uttid] = wavfp
                if uttid in filtered_uttids:
                    print(line.strip(), file=out_wav_f)

    # Filter the stm file
    stms = {}
    for file in os.listdir(stm_dir):
        fname = Path(file).stem
        stms[fname] = stm_dir / file
    for (fname, stm) in stms.items():
        with open(stm, 'r') as stm_f:
            with open(out_stm_dir / f"{fname}.stm", 'w') as out_stm_f:
                for i, line in tqdm(enumerate(stm_f)):
                    if i in filtered_stmidxs:
                        print(line.strip(), file=out_stm_f)

    # Filter the segid2stmidx file
    with open(out_segid2stmidx, 'w') as out_segid2stmidx_f:
        for segid, stmidx in segid2stmidx.items():
            if stmidx in filtered_stmidxs:
                print(f"{segid} {stmidx}", file=out_segid2stmidx_f)

    if dumpdir:
        # Write detailed info to the dumpdir for error analysis
        wer_dump = dumpdir / "wer.dump"
        download_dmp = dumpdir / "download.dump"
        spk_dump = dumpdir / "spk.dump"
        full_wer_dump = dumpdir / "full_wer.dump"
        mkdir_if_not_exist(wer_dump.parent)
        with open(wer_dump, 'w') as wer_dump_f:
            with open(download_dmp, 'w') as download_dump_f:
                with open(spk_dump, 'w') as spk_dump_f:
                    with open(full_wer_dump, 'w') as full_wer_dump_f:
                        # Categorize samples by wer with buckets ranging from 0.6 to 2.0, with a step of 0.2
                        wer_buckets = [0.6 + i * 0.2 for i in range(12)]
                        wer_buckets = [0.0] + wer_buckets + [2.0]
                        wer_buckets = [(wer_buckets[i], wer_buckets[i + 1])
                                    for i in range(len(wer_buckets) - 1)]
                        wer_buckets = {wer_bucket: [] for wer_bucket in wer_buckets}
                        for segid, wer in segid2wer.items():
                            for wer_bucket in wer_buckets:
                                if wer_bucket[0] <= wer < wer_bucket[1]:
                                    wer_buckets[wer_bucket].append((segid, wer))
                                    break
                        for wer_bucket, segids in wer_buckets.items():
                            print(f"WER bucket: {wer_bucket}", file=wer_dump_f)
                            # Shuffle the segids to avoid bias
                            random.shuffle(segids)
                            for segid, wer in segids[:50]:
                                uttid = "_".join(segid.split("_")[1:-1])
                                if wer <= 1.5 and wer > 0.4:
                                    print(f"{uttid2wavfp[uttid]}", file=download_dump_f)
                                print(f"{segid} {wer}", file=wer_dump_f)
                                print(
                                    f"REF_PHN:\t{segid2aligned[segid][0]}", file=wer_dump_f)
                                print(
                                    f"HYP_PHN:\t{segid2aligned[segid][1]}", file=wer_dump_f)
                                print(f"OP:\t\t{segid2aligned[segid][2]}", file=wer_dump_f)
                                print(f"CSID:\t{segid2aligned[segid][3]}", file=wer_dump_f)
                                print(f"REF:\t{segid2ref[segid]}", file=wer_dump_f)
                                print(f"TIME:\t{segid2time[segid]}", file=wer_dump_f)
                                print(f"WAV:\t{uttid2wavfp[uttid]}", file=wer_dump_f)
                                print(f"VAL:\t{segid in valid_segids}", file=wer_dump_f)
                                print("", file=wer_dump_f)
                                print(f"{segid} {wer}", file=full_wer_dump_f)

                        # Sort spk_valid_wer by the mean valid wer
                        spk_valid_wer = {spk: np.mean(np.array(wers))
                                        for spk, wers in spk_valid_wer.items()}
                        spk_valid_wer = list(
                            sorted(spk_valid_wer.items(), key=lambda x: x[1]))
                        for spk, val_wer in spk_valid_wer:
                            print(
                                f"{spk}\t{val_wer:.4f}\t{np.mean(np.array(spk_full_wer[spk])):.4f}\t{spk_duration[spk]:.2f}\t{spk_duration[spk] / 3600:.2f}\t{spk_duration[spk] / spk_uttnum[spk]:.2f}", file=spk_dump_f)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the text for Levenshtein distance computation.')
    parser.add_argument('-i', '--input_dir', type=Path, required=True,
                        help='The full path to the input directory to be filtered.')
    parser.add_argument('-o', '--output_dir', type=Path, required=True,
                        help='The full path to the output directory.')
    parser.add_argument('--wer', type=Path, required=True,
                        help='The full path to the wer file.')
    parser.add_argument('--aligned_file', type=Path, required=True,
                        help='The full path to the aligned file generated by the kaldi script.')
    parser.add_argument('--threshold', type=float, required=True,
                        help='The wer filter threshold (any instance with wer > threshold will be dumped).')
    parser.add_argument('--dumpdir', type=Path, default=None,
                        help='The full path to the dump directory for error analysis.')
    args = parser.parse_args()

    filter(input_dir=args.input_dir, output_dir=args.output_dir, aligned_file=args.aligned_file,
           wer=args.wer, threshold=args.threshold, dumpdir=args.dumpdir)


if __name__ == "__main__":
    main()
