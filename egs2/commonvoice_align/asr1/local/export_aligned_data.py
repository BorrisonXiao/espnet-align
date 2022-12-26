import argparse
import os
from pathlib import Path
from utils import scpid2mid, extract_mid, mkdir_if_not_exist, read_wavscp, read_utt2spk
from collections import defaultdict


def export_data(input_dir, output_dir, wavscp, utt2spk):
    # Build the wav.scp file
    wavs = read_wavscp(wavscp, raw=True)  # TODO: This might be bugged due to True, can be refined though
    utts = read_utt2spk(utt2spk, ignore_seg=True)
    with open(os.path.join(output_dir, "wav.scp"), "w") as f:
        with open(os.path.join(output_dir, "text_map"), "w") as ofh:
            with open(os.path.join(output_dir, "utt2spk"), "w") as utt_ofh:
                for file in os.listdir(input_dir):
                    uttid = Path(file).stem
                    print(uttid, wavs[uttid], file=f)
                    print(uttid, os.path.join(input_dir, file), file=ofh)
                    print(uttid, utts[uttid], file=utt_ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Break sentences for primary alignment results.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to decoded directory')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to output search map file')
    parser.add_argument('--wavscp', type=Path, required=True,
                        help='The full path to the wav.scp file')
    parser.add_argument('--utt2spk', type=Path, required=True,
                        help='The full path to the utt2spk file')
    args = parser.parse_args()

    export_data(input_dir=args.input_dir, output_dir=args.output_dir,
                wavscp=args.wavscp, utt2spk=args.utt2spk)


if __name__ == "__main__":
    main()
