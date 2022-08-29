from utils import Integerizer
import argparse
import os
from pathlib import Path
from utils import mkdir_if_not_exist
import pinyin_jyutping_sentence
import cn2an
import re
from seg2text import text_char_seg


def eliminate_tone(scp_phoneme):
    """
    Remove the trailing numbers representing tone information.
    e.g. "ji5" => "ji"
    """
    # res = re.sub(r"([a-z])\d", r"\1", scp_phoneme)
    return re.sub(r"([a-z])\d", r"\1", scp_phoneme)


def prep_txt(decoded_dir, text_map, output_dir, integerize=False, use_phoneme=False, ignore_tone=False):
    # Read the ground-truth text_map
    scps = {}
    with open(text_map, "r") as f:
        for line in f:
            uttid, scp_path = line.strip().split(" ", maxsplit=1)
            scps[uttid] = scp_path

    # Compute Levenshtein-distance based anchors for each utterance
    for decoded in os.listdir(decoded_dir):
        uttid = Path(decoded).stem
        if uttid not in scps:
            print(
                f"ERROR: Cannot find the {uttid}'s ground-truth text from {text_map}, either key error or the ground-truth text is not pre-processed.")
            exit(1)

        with open(os.path.join(decoded_dir, decoded), "r") as f:
            decoded_txt = f.read()
        with open(scps[uttid], "r") as f:
            scp = f.read()
            # Merge instances such as "1980 年" to "1980年" for better an2cn conversion
            scp = re.sub(r"(\d)\s年", r"\1年", scp)
            # Convert the numbers into characters and insert the spaces properly
            scp = text_char_seg(cn2an.transform(scp, "an2cn"))

        if integerize:
            txt_decoded, txt_scp = decoded_txt.strip().split(" "), scp.strip().split(" ")
            vocab = Integerizer(txt_decoded + txt_scp)
            decoded_txt, scp = " ".join(map(str, vocab.index(txt_decoded))), " ".join(
                map(str, vocab.index(txt_scp)))

        if use_phoneme:
            decoded_txt_phoneme = pinyin_jyutping_sentence.jyutping(
                decoded_txt, tone_numbers=True).replace("   ", " ")
            scp_phoneme = pinyin_jyutping_sentence.jyutping(
                scp, tone_numbers=True).replace("   ", " ")

            if ignore_tone:
                decoded_txt_phoneme = eliminate_tone(decoded_txt_phoneme)
                scp_phoneme = eliminate_tone(scp_phoneme)

            mkdir_if_not_exist(os.path.join(output_dir, "phoneme"))
            with open(os.path.join(output_dir, "phoneme", uttid + ".decoded"), "w") as f:
                print(f"{uttid} {decoded_txt_phoneme}", file=f)
            with open(os.path.join(output_dir, "phoneme", uttid + ".original"), "w") as f:
                print(f"{uttid} {scp_phoneme}", file=f)

        mkdir_if_not_exist(os.path.join(output_dir, "char"))
        with open(os.path.join(output_dir, "char", uttid + ".decoded"), "w") as f:
            print(f"{uttid} {decoded_txt}", file=f)
        with open(os.path.join(output_dir, "char", uttid + ".original"), "w") as f:
            print(f"{uttid} {scp}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Compute levenshtein distance between decoded and ground-truth text.')
    parser.add_argument('--decoded_dir', type=str, required=True,
                        help='The full path to the directory in which decoded text files are stored')
    parser.add_argument('--text_map', type=str, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which the comparison file will be stored')
    parser.add_argument('--integerize', action='store_true',
                        help='If option provided, the text will be integerized')
    parser.add_argument('--use_phoneme', action='store_true',
                        help='If option provided, the text will be converted to phoneme using pinyin_jyutping_sentence')
    parser.add_argument('--ignore_tone', action='store_true',
                        help='If option provided, the converted phonemes will disregard tones')
    args = parser.parse_args()

    prep_txt(args.decoded_dir, args.text_map, args.output_dir,
             args.integerize, args.use_phoneme, args.ignore_tone)


if __name__ == "__main__":
    main()
