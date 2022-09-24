import argparse
import os
from pathlib import Path
import subprocess
from ph2char import ph2char
import hashlib


def align_text(keyfile, eps, token_type, to_align_dir, raw_anchor_dir):
    """
    Perform text2text alignment of two text files by calling the align-text script in kaldi.
    """
    token_anchor_dir = os.path.join(raw_anchor_dir, token_type)
    char_anchor_dir = os.path.join(raw_anchor_dir, "char")
    with open(keyfile, "r") as f:
        for line in f:
            hyp, ref = line.strip().split(" ", maxsplit=1)
            # The filename will be hased to avoid the long filename error
            anchor_fname = "_vs_".join(
                [Path(hyp).stem, Path(ref).stem])
            fname_hash = hashlib.md5(anchor_fname.encode(
                'utf-8')).hexdigest() + ".anchor"
            anchor_fp = os.path.join(token_anchor_dir, fname_hash)
            with open(anchor_fp, "w") as ofh:
                print(anchor_fname, file=ofh)
            try:
                subprocess.call(f"align-text --special-symbol={eps} ark:{ref} ark:{hyp} ark,t:- \
                    | utils/scoring/wer_per_utt_details.pl --special-symbol={eps} >>{anchor_fp}", shell=True)

                if token_type == "phoneme":
                    hyp_char = hyp.replace("/phoneme/", "/char/")
                    ref_char = ref.replace("/phoneme/", "/char/")
                    char_anchor_fp = os.path.join(char_anchor_dir, fname_hash)
                    ph2char(anchor_file=anchor_fp, hyp_file=hyp_char,
                            ref_file=ref_char, output=char_anchor_fp, eps=eps)
            except subprocess.CalledProcessError as e:
                print(e.output)
                exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Compute the Levenshtein distance of two text files based on the given map file')
    parser.add_argument('--keyfile', type=Path, required=True,
                        help='The keyfile generated for alignment')
    parser.add_argument('--eps', type=str, default="---",
                        help='The epsilon string in the text')
    parser.add_argument('--token_type', choices=[
                        "char", "phoneme"], default="char", help='The type of token of the files to be aligned')
    parser.add_argument('--to_align_dir', type=Path, required=True,
                        help='The full path to the directory in which the to_align files are stored')
    parser.add_argument('--raw_anchor_dir', type=Path, required=True,
                        help='The full path to the directory in which the output anchor files will be stored')
    args = parser.parse_args()

    align_text(args.keyfile, args.eps, args.token_type,
               args.to_align_dir, args.raw_anchor_dir)


if __name__ == "__main__":
    main()
