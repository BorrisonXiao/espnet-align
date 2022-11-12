#!/usr/bin/env python3
# 2022 Cihan Xiao

import argparse
import pinyin_jyutping_sentence as pjs
from typing import TextIO
import re
import nltk

CONSONANTS = ["gw", "kw", "ng", "b", "c", "d", "f", "g", "h", "j", "k",
              "l", "m", "n", "p", "s", "t", "w", "z"]


def isEng(word):
    """
    Return true if the char is an English word.
    """
    return re.search('[a-zA-Z]', word) != None


def char2ph(input: TextIO, output: TextIO, tokenlist: bool = False):
    """
    Convert Cantonese chars to lexicons.
    """
    arpabet = nltk.corpus.cmudict.dict()
    if tokenlist:
        ools = set()
        lines = input.readlines()
        N = len(lines)
        print(f"{lines[0].strip()} {lines[0].strip()}", file=output)
        print(f"{lines[1].strip()} {lines[1].strip()}", file=output)
        for line in lines[2:-1]:
            char = line.strip()
            if not isEng(char):  # Avoid phonemize some english letters
                # Seems pjs has some bug that might produce unexpected "，"
                ph = re.sub("，", "", pjs.jyutping(char, tone_numbers=True))
                if ph != char:
                    for con in CONSONANTS:
                        if ph.startswith(con) and not ph[len(con)].isdigit():
                            ph = con + " " + ph[(len(con)):]
                            break  # Note that plural consonants are reached first
                    print(f"{char} {ph}", file=output)
                else:  # Those will be treated as OOLs since they are mostly puncs or rare Cantonese chars
                    ools.add(char)
                    print(f"{char} <ool>", file=output)
            else:
                # Use cmudict to phonemize English words
                char_lower = char.lower()
                # Note that OOVs will be characterized and converted
                ph = " ".join(arpabet[char_lower][0]).lower() if char_lower in arpabet else " ".join([
                    " ".join(arpabet[c][0]).lower() for c in char_lower])
                print(f"{char} {ph}", file=output)
        print(f"{lines[-1].strip()} {lines[-1].strip()}", file=output)

        print(
            f"There are {len(ools)} OOVs among {N} word types, i.e. OOV rate = {len(ools) / N * 100:.2f}%.")
        print(ools)
    else:
        raise NotImplementedError()


def main():
    parser = argparse.ArgumentParser(
        description='Convert Cantonese characters to jyutpings.')
    parser.add_argument('-i', '--input', type=argparse.FileType("r", encoding="utf-8"), required=True,
                        help='The full path to the input text in Cantonese.')
    parser.add_argument('-o', '--output', type=argparse.FileType("w", encoding="utf-8"), required=True,
                        help='The full path to the output lexicon file.')
    parser.add_argument('--tokenlist', action="store_true",
                        help='If specified, the input will be treated as a tokenlist file.')
    args = parser.parse_args()

    char2ph(args.input, args.output, args.tokenlist)


if __name__ == "__main__":
    main()
