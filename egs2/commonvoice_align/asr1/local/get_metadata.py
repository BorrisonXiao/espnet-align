#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
# from collections import defaultdict, Counter
from pathlib import Path
from utils import extract_mid, read_spkdump
import random
from collections import defaultdict

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
    spk2info = read_spkdump(input_dir / 'spk.dump')
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

    hklegco_features = output_dir / 'hklegco_features_spkmid'
    utt2spk = data_dir / 'asr' / 'utt2spk'
    segments = data_dir / 'asr' / 'segments'
    # For each utterance, extract the spkid and mid, then map the spkid to speaker gender
    with open(hklegco_features, 'w') as ofh:
        # with open(utt2spk, 'r') as f:
        #     lines = f.readlines()
        # # lines = random.sample(lines, k=1000)
        # for line in lines:
        #     line = line.strip()
        #     uttid, spkid = line.split()
        #     mid = extract_mid(uttid)
        #     gender = spkid2info[spkid]['gender']
        #     print(f"{uttid}\t{mid}\t{spkid}\t{gender}", file=ofh)

        # # Version 2: use spkid as the key. Note that the spkid entry is duplicated according to their total duration.
        # # Step 1: Get the mids associated with each spkid
        # spkid2mids = defaultdict(set)
        # with open(segments, 'r') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     line = line.strip()
        #     segid, uttid, _, _ = line.split()
        #     mid = extract_mid(uttid)
        #     spkid = segid.split('_')[0]
        #     spkid2mids[spkid].add(mid)
        # # Step 2: Write to the feature file
        # for spkid, info in spk2info.items():
        #     num_dups = int(max(info['duration'] // 3600, 1))
        #     gender = spkid2info[spkid]['gender']
        #     mids = ",".join(spkid2mids[spkid])
        #     for i in range(num_dups):
        #         print(f"{spkid}_{i+1}\t{mids}\t{gender}", file=ofh)

        # Version 3: Use spkid_mid as the key. This resolve the unsplitable issue.
        # e.g. A m1,m2
        #      B m1,m2
        # The case above is not splitable because the mids are not disjoint and the spkid is not duplicated.
        # A_m1 m1 0
        # A_m2 m2 2
        # B_m1 m1 3
        # B_m2 m2 1
        # The case above is splitable due to spk non-uniqueness.
        with open(utt2spk, 'r') as f:
            lines = f.readlines()
        # lines = random.sample(lines, k=1000)
        added = set()
        for line in lines:
            line = line.strip()
            uttid, spkid = line.split()
            mid = extract_mid(uttid)
            gender = spkid2info[spkid]['gender']
            key = f"{spkid}_{mid}"
            if key in added:
                continue
            print(f"{key}\t{mid}\t{spkid}\t{gender}", file=ofh)
            added.add(key)


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

    _metadata(input_dir=args.input_dir, data_dir=args.data_dir,
              output_dir=args.output_dir)


if __name__ == "__main__":
    main()
