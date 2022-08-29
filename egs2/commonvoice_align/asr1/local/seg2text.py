import logging
import argparse
from pathlib import Path
import os
import re


def text_char_seg(text):
    """
    Add space between chars as delimiter.
    Note that spaces won't be added between numbers and English letters,
    e.g. "1980" => "1980", "Financial Institution" => "Financial Institution",
    "一九八零年" => "一 九 八 零 年".
    """
    res = text
    # Replace spaces between English letters with "_", note that this should
    # be executed twice due to python re's implementation
    res = re.sub(r"([a-zA-Z\d]+)\s+([a-zA-Z\d]+)", r"\1_\2", res)
    res = re.sub(r"([a-zA-Z\d]+)\s+([a-zA-Z\d]+)", r"\1_\2", res)
    # Remove all spaces
    res = re.sub(r"\s+", "", res)
    # Replace the _ with a space
    res = re.sub(r"_", " ", res)
    # Add space after Chinese chars
    res = re.sub(r"([^a-zA-Z\d\s])", r"\1 ", res)
    # Add space between non-Chinese and Chinese chars
    res = re.sub(r"([a-zA-Z\d])([^a-zA-Z\d\s])", r"\1 \2", res)

    return res.strip()


def format_text(input_dir, text_output, utt2spk_output, char_seg):
    utt2spk = open(utt2spk_output, "w", encoding='utf-8')
    with open(text_output, "w", encoding='utf-8') as ofh:
        for seg_file in os.listdir(input_dir):
            with open(os.path.join(input_dir, seg_file), "r", encoding='utf-8') as f:
                for line in f:
                    segid, uttid, _, _, _, text = line.strip().split(" ", maxsplit=5)
                    # TODO: Char level tokenization
                    if char_seg:
                        text = text_char_seg(text)
                    print(f"{segid}_{uttid} {text}", file=ofh)

                    # Currently using metadata as speaker ID
                    spk = uttid.split("_")[-1]
                    print(f"{segid}_{uttid} {spk}", file=utt2spk)
    utt2spk.close()


def main():
    parser = argparse.ArgumentParser(
        description='Process segmentation files and create a text file')
    parser.add_argument('--seg_file_dir', type=Path, required=True,
                        help='Input: The directory with segment files.')
    parser.add_argument('--text_output', type=Path, required=True,
                        help='Output: The file to store the formatted text file.')
    parser.add_argument('--utt2spk_output', type=Path, required=True,
                        help='Output: The file to store the formatted utt2spk file.')
    parser.add_argument('--char_seg', action='store_true',
                        help='If option provided, perform the char-level tokenization (for Cantonese)')
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    format_text(args.seg_file_dir, args.text_output, args.utt2spk_output, args.char_seg)


if __name__ == "__main__":
    main()
