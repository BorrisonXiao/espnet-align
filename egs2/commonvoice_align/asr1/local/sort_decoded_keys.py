import argparse
import os
from pathlib import Path


def generate_keys(decoded_dir, output):
    res = []
    for file in os.listdir(decoded_dir):
        uttid = Path(file).stem
        res.append((uttid, os.path.join(decoded_dir, file)))

    # First sort by utt order
    res = sorted(res, key=lambda x: int(x[0].split("_")[2]))
    # Then sort by mid
    res = sorted(res, key=lambda x: x[0].split("_")[1])

    with open(output, "w") as f:
        for (uttid, fp) in res:
            print(uttid, fp, file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Sort the decoded hypothesis scripts in chronological order.')
    parser.add_argument('--decoded_dir', type=Path, required=True,
                        help='The full path to the directory in which decoded text files are stored')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the output key file')
    args = parser.parse_args()

    generate_keys(decoded_dir=args.decoded_dir, output=args.output)


if __name__ == "__main__":
    main()
