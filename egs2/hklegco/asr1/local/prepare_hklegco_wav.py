import re
import argparse
import os
from pathlib import Path
import subprocess


def get_mid_from_path(input_filename):
    mid_regex = re.compile(r"(video\/)(M.*)(\/can)")
    mid = mid_regex.search(input_filename).group(2)
    return mid


def generate_scp(input_dir, output_dir):
    for file in os.listdir(input_dir):
        fname = Path(file).stem
        full_path = os.path.join(input_dir, file)
        uttid = f"{get_mid_from_path(full_path)}_{fname}"
        subprocess.call(["ffmpeg", "-i", full_path, "-f", "wav", "-ar", "16000",
                        "-ab", "16", "-ac", "1", os.path.join(output_dir, uttid + ".wav")])


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the HKLEGCO audio data')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input: The directory containing the original .mp3 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output: The directory containing the processed .wav files.')
    args = parser.parse_args()

    generate_scp(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
