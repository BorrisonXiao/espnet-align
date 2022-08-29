import argparse
import os
from pathlib import Path


def generate_clip_info(input_dir, output, input_format):
    with open(output, "w") as f:
        for uttid in os.listdir(input_dir):
            clip_fp = os.path.join(input_dir, uttid, f"{uttid}.json") if input_format == "json" else os.path.join(
                input_dir, uttid, f"segments")
            if not os.path.exists(clip_fp):
                raise ValueError(f"The clip file is not found for {uttid}")
            print(f"{uttid} {clip_fp}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate segment key files in the format "<uttid>" "<clip_file_path>".')
    parser.add_argument('--vad_dir', type=Path, required=True,
                        help='The full path to the directory in which the VAD segmentation info files are stored')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the key file in which the result will be stored')
    parser.add_argument('--input_format', required=True, choices=["json", "kaldi"],
                        help='The format of the input clip_info file, either "json" or "kaldi"')
    args = parser.parse_args()

    generate_clip_info(input_dir=args.vad_dir, output=args.output,
                       input_format=args.input_format)


if __name__ == "__main__":
    main()
