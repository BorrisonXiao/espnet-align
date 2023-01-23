#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
# from collections import defaultdict, Counter
from pathlib import Path
from utils import extract_mid
import random

random.seed(0)

GENDER = ['Female', 'Male', 'Unknown']


def read_annot(file):
    spkid2info = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        splitted = line.split('\t')
        spkid, val_wer, full_wer, secs, hrs, avg_secs, gender = splitted
        spkid2info[spkid] = dict(val_wer=float(val_wer), full_wer=float(
            full_wer), secs=float(secs), hrs=float(hrs), avg_secs=float(avg_secs), gender=int(gender))
    return spkid2info


def _metadata(input_dir, data_dir, output_dir):
    annot_file = input_dir / 'spk.annot.dump'
    spkid2info = read_annot(annot_file)
    spk_num = len(spkid2info)
    total_len = 0
    gender_info = [dict(time=0, count=0) for _ in range(len(GENDER))]
    for spkid, info in spkid2info.items():
        gender_info[info['gender']]['count'] += 1
        gender_info[info['gender']]['time'] += info['secs']
        total_len += info['secs']
    # Python dicts are insertion-ordered since Python 3.7
    output = output_dir / 'metadata.dump'
    with open(output, 'w') as f:
        # Write the gender ratio
        for gender, info in enumerate(gender_info):
            print(f"{GENDER[gender]}: {info['count']}/{spk_num}={info['count']/spk_num:.2f} {info['time']/3600:.2f}/{total_len/3600:.2f}={info['time']/total_len:.2f}", file=f)
    
    hklegco_features = output_dir / 'hklegco_features'
    utt2spk = data_dir / 'asr' / 'utt2spk'
    # For each utterance, extract the spkid and mid, then map the spkid to speaker gender
    with open(hklegco_features, 'w') as ofh:
        with open(utt2spk, 'r') as f:
            lines = f.readlines()
        # lines = random.sample(lines, k=1000)
        for line in lines:
            line = line.strip()
            uttid, spkid = line.split()
            mid = extract_mid(uttid)
            gender = spkid2info[spkid]['gender']
            print(f"{uttid}\t{mid}\t{spkid}\t{gender}", file=ofh)



def main():
    parser = argparse.ArgumentParser(
        description='Get some metadata information from the annotated spk file.')
    parser.add_argument('-i', '--input_dir', type=Path, required=True,
                        help='The full path to the input directory to be splitted.')
    parser.add_argument('-d', '--data_dir', type=Path, required=True,
                        help='The full path to the data directory.')
    parser.add_argument('-o', '--output_dir', type=Path, required=True,
                        help='The full path to the output file.')
    args = parser.parse_args()

    _metadata(input_dir=args.input_dir, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
