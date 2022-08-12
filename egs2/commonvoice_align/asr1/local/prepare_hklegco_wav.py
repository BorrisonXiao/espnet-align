import re
import argparse
import os
from pathlib import Path


def get_mid_from_path(input_filename):
    mid_regex = re.compile(r"(video\/)(M.*)(\/can)")
    mid = mid_regex.search(input_filename).group(2)
    return mid


def generate_scp(input_dir, output_filename):
    ofh = open(output_filename, 'w', encoding='utf-8')
    for file in os.listdir(input_dir):
        fname = Path(file).stem
        full_path = os.path.join(input_dir, file)
        uttid = f"{get_mid_from_path(full_path)}_{fname}"
        print(f"{uttid} ffmpeg -i {full_path} -f wav -ar 16000 -ab 16 -ac 1 - |", file=ofh)
    ofh.close()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the HKLEGCO audio data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input: The directory containing the original .mp3 files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output: The generated wav.scp file, etc.')
    args = parser.parse_args()

    generate_scp(args.input, args.output)


if __name__ == "__main__":
    main()
