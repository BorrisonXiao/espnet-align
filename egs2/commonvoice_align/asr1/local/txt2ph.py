#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
from tqdm import tqdm
from pathlib import Path
import pinyin_jyutping_sentence
import re


def eliminate_tone(scp_phoneme):
    """
    Remove the trailing numbers representing tone information.
    e.g. "ji5" => "ji"
    """
    return re.sub(r"([a-z])\d", r"\1", scp_phoneme)


def txt2ph(file, output):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(output, 'w') as f:
        for line in tqdm(lines):
            line = line.strip()
            splitted = line.split(maxsplit=1)
            # Handle empty lines
            if len(splitted) != 2:
                sentid = splitted[0]
                print(f"{sentid}", file=f)
                continue
            sentid, text = splitted
            text = re.sub(r'\s+', ' ', text)
            text = pinyin_jyutping_sentence.jyutping(
                text, tone_numbers=True).replace("   ", " ")
            text = eliminate_tone(text)
            print(f"{sentid} {text}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the text for Levenshtein distance computation.')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='The full path to the input file to be converted.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='The full path to the output file.')
    args = parser.parse_args()

    txt2ph(file=args.input, output=args.output)


if __name__ == "__main__":
    main()
