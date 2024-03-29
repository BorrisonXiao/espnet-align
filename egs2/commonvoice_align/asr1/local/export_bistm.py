#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import extract_mid, read_raw_stm, mkdir_if_not_exist, read_t2t, cut_uttid_len
import json
from tqdm import tqdm
from collections import defaultdict
import logging


def parse_t2t(t2t_file):
    """
    Parse the text2text file.
    """
    # Read the text2text file based on the date
    assert os.path.exists(t2t_file), f"{t2t_file} does not exist."
    t2t = read_t2t(t2t_file)

    # Parse the t2t results so that each idx maps to a line in the t2t file
    t2t_idx2line = {}
    for line in t2t:
        cn_idxs = line[2]
        for idx in cn_idxs:
            t2t_idx2line[idx] = line

    return t2t_idx2line


def export_bistm(rstm_dir, wavscp, punc_text_dir, datefile, idx_dir, output_dir, t2t_dir, score_threshold):
    """
    Export the text in the stm file with its corresponding index in the meeting level scp.
    """
    mkdir_if_not_exist(output_dir)

    # Read the wav.scp file
    uttid2fp = {}
    with open(wavscp, 'r') as f:
        for line in f:
            uttid, fp = line.strip().split(maxsplit=1)
            uttid2fp[uttid] = fp

    # Read the date file
    with open(datefile, 'r') as f:
        mid2date = json.load(f)

    # Organize the rstm files by date so that the t2t file can be parsed only once
    date2rstm = defaultdict(list)
    for rstm_file in os.listdir(rstm_dir):
        uttid = Path(rstm_file).stem
        mid = extract_mid(uttid)
        date = mid2date[mid]
        date2rstm[date].append(rstm_file)

    # Read punctuation text if it exists
    if punc_text_dir:
        mid2punctxt = {}
        for punctxt_file in os.listdir(punc_text_dir):
            mid = Path(punctxt_file).stem
            with open(punc_text_dir / punctxt_file, 'r') as f:
                lines = f.readlines()
                mid2punctxt[mid] = [line.strip().split(maxsplit=1)[1] for line in lines]

    # Create directory for placing the st data
    st_dir = os.path.join(output_dir, 'st')
    mkdir_if_not_exist(st_dir)

    # Create directory for placing the asr data
    asr_dir = os.path.join(output_dir, 'asr')
    mkdir_if_not_exist(asr_dir)

    # Create a metadata directory for placing the segid2stmidx file
    metadata_dir = os.path.join(output_dir, 'metadata')
    mkdir_if_not_exist(metadata_dir)

    added_uttids = set()

    with open(os.path.join(st_dir, 'st-can2eng.all.stm'), 'w') as ofh:
        with open(os.path.join(st_dir, 'asr-can.all.stm'), 'w') as asr_ofh:
            with open(os.path.join(asr_dir, 'wav.scp'), 'w') as wav_ofh:
                with open(os.path.join(asr_dir, 'utt2spk'), 'w') as utt2spk_ofh:
                    with open(os.path.join(asr_dir, 'text'), 'w') as text_ofh:
                        with open(os.path.join(asr_dir, 'segments'), 'w') as seg_ofh:
                            with open(os.path.join(metadata_dir, 'segid2stmidx'), 'w') as segid2stmidx_ofh:
                                stmidx = 0
                                # Iterate over the rstm files
                                for date, rstm_files in date2rstm.items():
                                    t2t_file = os.path.join(t2t_dir, f'{date}.t2t')
                                    t2t_idx2line = parse_t2t(t2t_file)
                                    for rstm_file in tqdm(rstm_files):
                                        uttid = Path(rstm_file).stem
                                        rstm = read_raw_stm(
                                            os.path.join(rstm_dir, rstm_file))
                                        idx_file = os.path.join(
                                            idx_dir, f'{uttid}.rstm.idx')
                                        assert os.path.exists(
                                            idx_file), f"{idx_file} does not exist."

                                        # Read the index file
                                        with open(idx_file, 'r') as f:
                                            idxs = [int(line.strip()) for line in f]

                                        # BUGFIX: multiple indices for the can text will be merged
                                        next_line = None
                                        next_can_text = None
                                        merge = False

                                        for i, idx in enumerate(idxs):
                                            # Get the corresponding line in the t2t file
                                            try:
                                                t2t_line = t2t_idx2line[idx]
                                            except KeyError as e:
                                                logging.warning(
                                                    "The index does not exist in the t2t file.")
                                                logging.warning(date)
                                                next_line = None
                                                next_can_text = None
                                                merge = False
                                                continue

                                            # Update the next lines
                                            if i < len(idxs) - 1:
                                                try:
                                                    # Update the next English line
                                                    next_line = t2t_idx2line[idxs[i + 1]]
                                                except KeyError as e:
                                                    logging.warning(
                                                        "The index does not exist in the t2t file.")
                                                    logging.warning(date)
                                                    next_line = None
                                                    next_can_text = None
                                                # Update the next Cantonese text
                                                next_can_text = rstm[i + 1][4]
                                            else:
                                                next_line = None
                                                next_can_text = None

                                            if merge:
                                                merge = False
                                                # The Cantonese text will be appended
                                                can_text += rstm[i][4] if not punc_text_dir else mid2punctxt[extract_mid(
                                                    uttid)][idx]
                                            else:
                                                # Get the start time
                                                start_time = rstm[i][2]
                                                # Get the cantonese text
                                                can_text = rstm[i][4] if not punc_text_dir else mid2punctxt[extract_mid(
                                                    uttid)][idx]

                                            # The end time will be updated anyway
                                            end_time = rstm[i][3]
                                            # Merge the data if the content is the same as the next one, note there exsits a rare case (ASR error) that the same cantonese text is repeated twice
                                            if i != len(idxs) - 1 and next_line == t2t_line and next_can_text != rstm[i][4]:
                                                assert len(t2t_line[-2]) > 1 or i == len(
                                                    idxs) - 1, f"The length of the indices should be greater than 1, instead it is {t2t_line} with i = {i} and {len(idxs) - 1}, \nprev is {next_line}\nin rstm file: {os.path.join(rstm_dir, rstm_file)}"
                                                merge = True
                                                continue

                                            # Get the score
                                            score = t2t_line[1]
                                            # Skip the line if the score is above the threshold or if the score is 0 (i.e. the alignment is empty)
                                            if score > score_threshold or score == 0:
                                                next_line = None
                                                next_can_text = None
                                                merge = False
                                                continue
                                            # Get the English text
                                            eng_text = t2t_line[-1]
                                            # Get the spkid
                                            spkid = t2t_line[0]
                                            # Get the uttid and sentid, note that upon a merge, the sentid will be overwritten by the later one
                                            sentid, uttid = rstm[i][0], rstm[i][1]
                                            # Get the audio file path
                                            fp = uttid2fp[uttid]
                                            # Write the line to the output stm files
                                            line = f"{fp} 1 {spkid} {start_time} {end_time} {eng_text}"
                                            print(line, file=ofh)
                                            asr_line = f"{fp} 1 {spkid} {start_time} {end_time} {can_text}"
                                            print(asr_line, file=asr_ofh)

                                            # Trim the uttid
                                            uttid = cut_uttid_len(uttid, 64, 64)

                                            if uttid not in added_uttids:
                                                added_uttids.add(uttid)
                                                # Write the (uttid, filepointer) to the wav.scp file
                                                print(f"{uttid} {fp}", file=wav_ofh)

                                            # Write the (segid, uttid, start_time, end_time) to the segments file
                                            segid = f"{spkid}_{uttid}_{sentid}"
                                            print(
                                                f"{segid} {uttid} {start_time} {end_time}", file=seg_ofh)
                                            # Write the (segid, text) to the text file
                                            print(f"{segid} {can_text}", file=text_ofh)
                                            # Write the (segid, spkid) to the utt2spk file
                                            print(f"{segid} {spkid}", file=utt2spk_ofh)

                                            # Write the (segid, stm_idx) to the segid2stmidx file
                                            print(f"{segid} {stmidx}",
                                                  file=segid2stmidx_ofh)
                                            stmidx += 1


def main():
    """
    Make the bilingual stm file and the kaldi-style asr files.
    """
    logging.basicConfig(
        level="INFO", format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Make the bilingual stm file and the kaldi-style asr files.')
    parser.add_argument('--rstm_dir', type=Path, required=True,
                        help='The full path to the directory in which the audio-text .stm files are stored.')
    parser.add_argument('--punc_text_dir', type=Path, default=None,
                        help='The full path to the directory in which the punctuation text are stored.')
    parser.add_argument('--wavscp', type=Path, required=True,
                        help='The wav.scp file pointing to the source audio files.')
    parser.add_argument('--datefile', type=Path, required=True,
                        help='The json date file converting mid to date.')
    parser.add_argument('--idx_dir', type=Path, required=True,
                        help='The full path to the directory in which the absolute line index of the audio-text alignment results w.r.t the transcript are stored.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--t2t_dir', type=Path, required=True,
                        help='The t2t_dir storing the text-to-text alignment results (with ambiguous spkid), i.e. the .t2t files.')
    parser.add_argument('--score_threshold', type=float, default=0.627,
                        help='The score threshold for filtering bad t2t alignment results.')
    args = parser.parse_args()
    export_bistm(rstm_dir=args.rstm_dir, wavscp=args.wavscp, idx_dir=args.idx_dir, punc_text_dir=args.punc_text_dir,
                 output_dir=args.output_dir, t2t_dir=args.t2t_dir, datefile=args.datefile, score_threshold=args.score_threshold)


if __name__ == "__main__":
    main()
