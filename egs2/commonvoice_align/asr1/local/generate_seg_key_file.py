import argparse
import os
from pathlib import Path
from utils import read_anchor_file, segid2uttid, mkdir_if_not_exist


def clear_buffer(buffer, output_dir):
    for uttid, txts in buffer.items():
        txt = "\n".join(txts)
        merged_dir = os.path.join(output_dir, uttid)
        mkdir_if_not_exist(merged_dir)
        merged_fname = os.path.join(merged_dir, "text")
        with open(merged_fname, "a") as ofh:
            print(txt, file=ofh)


def dir2key(decoded_dir, anchor_dir, scp_map, output):
    uttid2scpid = {}
    with open(scp_map, "r") as f:
        for line in f:
            uttid, scpid = line.strip().split(maxsplit=1)
            uttid2scpid[uttid] = scpid

    with open(output, "w") as f:
        for anchor_fname in os.listdir(anchor_dir):
            anchor_file = os.path.join(anchor_dir, anchor_fname)
            fname, _, _, _, _ = read_anchor_file(anchor_file)
            uttid, scpid = fname.split("_vs_")
            # If the uttid is not found in the map, there is no match for the audio and thus skipped
            # Also skip if the anchor file is not the correct mapping in the scp_map file
            if uttid not in uttid2scpid or uttid2scpid[uttid] != scpid:
                continue
            textfile = os.path.join(decoded_dir, uttid, "text")
            if not os.path.exists(textfile):
                raise ValueError(f"Decoded text file not found at {textfile}")

            print(
                f"{os.path.abspath(anchor_file)} {os.path.abspath(textfile)}", file=f)


def text2key(text, anchor_dir, scp_map, output, max_line_cache=1000):
    output_dir = os.path.join(Path(output).parent.absolute(), "dump")
    # Separate each utterance's VAD segments into a single file
    with open(text, "r", encoding="utf-8") as f:
        buffer = {}
        counter = 0
        for line in f:
            splitted = line.strip().split(" ", maxsplit=1)
            if len(splitted) <= 1:
                continue
            segid, txt = splitted
            uttid = segid2uttid(segid)
            if uttid not in buffer:
                buffer[uttid] = [f"{segid} {txt}"]
            else:
                buffer[uttid].append(f"{segid} {txt}")
            counter += 1

            # Clear buffer and write to file
            if counter == max_line_cache:
                clear_buffer(buffer=buffer, output_dir=output_dir)
                counter = 0
                buffer = {}
        clear_buffer(buffer=buffer, output_dir=output_dir)

    dir2key(output_dir, anchor_dir, scp_map, output)


def main():
    parser = argparse.ArgumentParser(
        description='Generate segment key files in the format "<anchor_file_path>" "<text_file_path>".')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--decoded_dir",
        default=None,
        type=Path,
        help="The full path to the directory in which decoded text files are stored",
    )
    group.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Path to the text file that stores the decoded text"
    )
    parser.add_argument('--anchor_dir', type=Path, required=True,
                        help='The full path to the directory in which the anchor files are stored')
    parser.add_argument('--scp_map', type=Path, required=True,
                        help='The full path to the file in which the mapping of audio uttid and text scpid is stored')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the key file in which the segmentation files will be stored')
    args = parser.parse_args()

    if args.decoded_dir:
        dir2key(args.decoded_dir, args.anchor_dir,
                scp_map=args.scp_map, output=args.output)
    elif args.text:
        text2key(args.text, args.anchor_dir, scp_map=args.scp_map, output=args.output)


if __name__ == "__main__":
    main()
