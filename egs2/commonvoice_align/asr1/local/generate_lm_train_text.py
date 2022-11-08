import argparse
import os
from pathlib import Path
from utils import mkdir_if_not_exist
import cn2an
import re
from seg2text import text_char_seg, _add_space_digits


def generate_text(text_map, output_dir, sentence):
    with open(text_map, "r") as f:
        for line in f:
            uttid, txt_path = line.strip().split(" ", maxsplit=1)

            txt_dir = os.path.join(output_dir, uttid)
            mkdir_if_not_exist(txt_dir)

            if sentence:
                with open(txt_path, "r") as fp:
                    text = fp.readlines()

                text = [re.sub(r"(\d\d\d\d)\s年", r"\1年", t.strip()) for t in text]
                with open(os.path.join(txt_dir, "lm_train.txt"), "w") as ofh:
                    for t in text:
                        sentid, scp = t.strip().split(maxsplit=1)
                        # Break long numbers into digits
                        scp = re.sub(r"(\d{16,})", _add_space_digits, scp)
                        scp = text_char_seg(cn2an.transform(scp, "an2cn"))
                        print(f"{sentid} {scp}", file=ofh)
            else:
                with open(txt_path, "r") as fp:
                    text = fp.read()

                text = re.sub(r"(\d\d\d\d)\s年", r"\1年", text)
                # Convert the numbers into characters and insert the spaces properly
                text = text_char_seg(cn2an.transform(text, "an2cn"))

                with open(os.path.join(txt_dir, "lm_train.txt"), "w") as ofh:
                    print(f"{uttid} {text}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Generate text files for each utterance that will be aligned.')
    parser.add_argument('--text_map', type=Path, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the text file will be stored')
    parser.add_argument('--sentence', action='store_true',
                        help='If true, the lm_train.txt will be sentence-level')
    args = parser.parse_args()

    generate_text(args.text_map, args.output_dir, args.sentence)


if __name__ == "__main__":
    main()
