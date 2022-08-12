import logging
import argparse
from pathlib import Path
import os


def segid2uttid(segid):
    splitted = segid.split("_")
    return "_".join(splitted[:-1])


def clear_buffer(buffer, output_dir):
    for uttid, txt in buffer.items():
        txt = txt.strip()
        merged_fname = os.path.join(output_dir, uttid + ".txt")
        if os.path.exists(merged_fname):
            with open(merged_fname, "a") as ofh:
                ofh.write(" " + txt)
        else:
            with open(merged_fname, "a") as ofh:
                ofh.write(txt)


def merge_decode_text(input, output_dir, max_line_cache=1000):
    with open(input, "r", encoding='utf-8') as f:
        buffer = {}
        counter = 0
        for line in f:
            segid, text = line.strip().split(" ", maxsplit=1)
            uttid = segid2uttid(segid)
            buffer[uttid] = text if uttid not in buffer else " ".join(
                [buffer[uttid], text])
            counter += 1

            # Clear buffer and write to file
            if counter == max_line_cache:
                clear_buffer(buffer=buffer, output_dir=output_dir)
                counter = 0
                buffer = {}
        clear_buffer(buffer=buffer, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Process segmentation files and create a text file')
    parser.add_argument('--input', type=Path, required=True,
                        help='The full path to the decoded text file or the text.key file storing the groud-truth scripts.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The directory to store the output decode_text file for each utt.')
    parser.add_argument('--decode',
                        action='store_true',
                        help='If option provided, the input file format is treated as decoded text.')
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    merge_decode_text(args.input, args.output_dir)


if __name__ == "__main__":
    main()
