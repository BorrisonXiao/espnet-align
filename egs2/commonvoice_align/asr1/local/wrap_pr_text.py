import os
import argparse
from pathlib import Path


def wrap_pr_text(text_map, dump_dir, scp_map, wav_scp, output_dir):
    """
    Wrap up the text file and form a kaldi text file based on primary alignment results.
    """
    utt2spk = os.path.join(output_dir, "utt2spk.unsorted")
    text_output = os.path.join(output_dir, "text.unsorted")
    wav_scp_output = os.path.join(output_dir, "wav.scp.unsorted")
    if text_map:
        scpid2scpfp = {}  # Stores the {scpid: scp_file_pointer} pair
        with open(text_map, "r") as f:
            for line in f:
                line = line.strip()
                scpid, scpfp = line.split(maxsplit=1)
                scpid2scpfp[scpid] = scpfp

    uttids = {}
    with open(utt2spk, "w") as fp:
        with open(text_output, "w") as ofh:
            scpid2uttid = {}
            with open(scp_map, "r") as f:
                for line in f:
                    uttid, scpid = line.strip().split(maxsplit=1)
                    spkid = scpid.split('_')[0]
                    uttids[uttid] = spkid
                    # The spkid must be the prefix to keep kaldi happy
                    uttid = "_".join([spkid, uttid])
                    # Generate new text from the original reference script with arabic nums, etc.
                    if text_map:
                        with open(scpid2scpfp[scpid], "r") as txtfp:
                            text = txtfp.read().strip()
                        print(f"{uttid} {text}", file=ofh)
                    else:
                        scpid2uttid[scpid] = uttid
                    print(f"{uttid} {spkid}", file=fp)
            # Generate new text from the normalized text reference scripts
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

    with open(wav_scp, "r") as f:
        with open(wav_scp_output, "w") as ofh:
            for line in f:
                uttid, rest = line.strip().split(maxsplit=1)
                if uttid in uttids:
                    print(f"{uttids[uttid]}_{uttid} {rest}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Determine anchor utterances based on the aligned text file.')
    parser.add_argument('--scp_map', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    parser.add_argument('--wav_scp', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text_map", type=Path)
    group.add_argument("--dump_dir", type=Path,
                       help="Dump directory with normalized text")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    args = parser.parse_args()

    wrap_pr_text(args.text_map, args.dump_dir, args.scp_map,
                 args.wav_scp, args.output_dir)


if __name__ == "__main__":
    main()
