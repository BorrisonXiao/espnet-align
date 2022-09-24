import os
import argparse
from pathlib import Path


def wrap_pr_text(text_map, dump_dir, scp_map, output, utt2spk):
    """
    Wrap up the text file and form a kaldi text file based on primary alignment results.
    """
    if text_map:
        scpid2scpfp = {}  # Stores the {scpid: scp_file_pointer} pair
        with open(text_map, "r") as f:
            for line in f:
                line = line.strip()
                scpid, scpfp = line.split(maxsplit=1)
                scpid2scpfp[scpid] = scpfp

    with open(utt2spk, "w") as fp:
        with open(output, "w") as ofh:
            scpid2uttid = {}
            with open(scp_map, "r") as f:
                for line in f:
                    uttid, scpid = line.strip().split(maxsplit=1)
                    if text_map:
                        with open(scpid2scpfp[scpid], "r") as txtfp:
                            text = txtfp.read().strip()
                        print(f"{uttid} {text}", file=ofh)
                    else:
                        scpid2uttid[scpid] = uttid
                    print(f"{uttid} {scpid.split('_')[0]}", file=fp)
            if dump_dir:
                for mid in os.listdir(dump_dir):
                    char_dir = os.path.join(dump_dir, mid, "ref", "char")
                    for scp_fname in os.listdir(char_dir):
                        scpid = Path(scp_fname).stem
                        if scpid in scpid2uttid:
                            with open(os.path.join(char_dir, scp_fname), "r") as txtfp:
                                text = txtfp.read().strip()
                            print(
                                f"{scpid2uttid[scpid]} {text.split(maxsplit=1)[-1]}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Determine anchor utterances based on the aligned text file.')
    parser.add_argument('--scp_map', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text_map", type=Path)
    group.add_argument("--dump_dir", type=Path,
                       help="Dump directory with normalized text")
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    parser.add_argument('--utt2spk', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    args = parser.parse_args()

    wrap_pr_text(args.text_map, args.dump_dir, args.scp_map, args.output, args.utt2spk)


if __name__ == "__main__":
    main()
