#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
from utils import mkdir_if_not_exist
import shutil


def _seg(segments, output, vad=True, window_size=180, overlap=30):
    """
    Sliding window resegmentation.
    The full-text flexible alignment will be performed on each individual
    re-segged segment.
    """
    with open(segments, "r") as f:
        lines = f.readlines()
    with open(output, "w") as ofh:
        segid = 0
        if vad:
            for i, line in enumerate(lines):
                try:
                    _, uttid, start, end = line.strip().split()
                    start, end = float(start), float(end)
                except:
                    raise ValueError(f"Error: Broken segments file {segments}...")
                if i == 0:
                    seg_start = start
                if end >= seg_start + window_size or i == len(lines) - 1:
                    seg_end = end
                    print(f"{uttid}_seg{segid:04}", uttid, seg_start, seg_end, file=ofh)
                    # Doesn't really need the start to be vad segmented
                    seg_start = seg_end - overlap
                    segid += 1
        else:
            _, uttid, _, audio_len = lines[-1].strip().split()
            audio_len = float(audio_len)
            seg_start, seg_end = 0, 0
            while seg_end < audio_len:
                seg_end = min(seg_start + window_size, audio_len)
                print(f"{uttid}_seg{segid:04}", uttid, seg_start, seg_end, file=ofh)
                seg_start = seg_end - overlap
                segid += 1


def reseg(wav_scp, output_dir, vad_dir, vad=True, window_size=180, overlap=30):
    uttids = []
    raw_dir = os.path.join(output_dir, "decode")
    mkdir_if_not_exist(raw_dir)
    with open(wav_scp, "r") as f:
        for line in f:
            uttids.append(line.strip().split(maxsplit=1)[0])
    for uttid in uttids:
        segfile = os.path.join(vad_dir, "decode", uttid, "segments")
        assert os.path.exists(segfile)
        _out_dir = os.path.join(raw_dir, uttid)
        mkdir_if_not_exist(_out_dir)

        # Copy over the wav.scp in vad_dir
        src_scp = os.path.join(vad_dir, "decode", uttid, "wav.scp")
        dst_scp = os.path.join(_out_dir, "wav.scp")
        shutil.copyfile(src_scp, dst_scp)

        output = os.path.join(_out_dir, "segments")
        _seg(segfile, output, vad=vad, window_size=window_size, overlap=overlap)


def main():
    """
    Perform resegmentation (based on sliding windows) for flexible alignment.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Perform resegmentation (based on sliding windows) for flexible alignments.')
    parser.add_argument('--keyfile', type=Path, required=True,
                        help='The full path to the keyfile, e.g. wav.scp.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('--vad_dir', type=Path, required=True,
                        help='Path to the vad directory.')
    parser.add_argument('--vad', action="store_true",
                        help='Whether to refer to vad segments.')
    parser.add_argument('--overlap_size', type=float, default=30,
                        help='The size of the overlap between windows.')
    parser.add_argument('--window_size', type=float, default=180,
                        help='The size of the resegmentation windows.')
    args = parser.parse_args()
    reseg(wav_scp=args.keyfile,
          output_dir=args.output_dir, vad_dir=args.vad_dir, vad=args.vad, window_size=args.window_size, overlap=args.overlap_size)


if __name__ == "__main__":
    main()
