import argparse
import os
from pathlib import Path


def mkdir_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def segid2uttid(segid):
    return "_".join(segid.split("_")[:-1])


def split_scps(scp, output_dir):
    with open(scp, "r") as f:
        for line in f:
            segid, wav_path = line.strip().split(" ", maxsplit=1)
            uttid = segid2uttid(segid)

            wav_dir = os.path.join(output_dir, uttid)
            mkdir_if_not_exist(wav_dir)

            wav_scp = os.path.join(wav_dir, "wav.scp")
            with open(wav_scp, "a") as ofh:
                print(f"{segid} {wav_path}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Generate text files for each utterance that will be aligned.')
    parser.add_argument('--input', type=Path, required=True,
                        help='The full path to the wav.scp file')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the splitted wav.scp files will be stored')
    args = parser.parse_args()

    split_scps(args.input, args.output_dir)


if __name__ == "__main__":
    main()
