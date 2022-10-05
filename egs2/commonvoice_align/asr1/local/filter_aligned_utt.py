import argparse
from pathlib import Path
import os
from utils import segid2uttid


def filter_aligned_utt(scp_map, wav_scp, output_dir):
    """
    Filter the wav.scp file from based on the primary alignment result to form new dataset.
    """
    output = os.path.join(output_dir, "wav.scp.unsorted")
    utt2spk = os.path.join(output_dir, "utt2spk.unsorted")
    text_raw = os.path.join(output_dir, "text.raw")
    text = os.path.join(output_dir, "text.unsorted")
    uttid2spkid = {}  # Stores the {uttid: spkid} pair
    segids = set()  # Stores the aligned segments' segid
    with open(scp_map, "r") as f:
        for line in f:
            uttid, scpid = line.strip().split(maxsplit=1)
            # The spkid by default is the first part of the scpid
            uttid2spkid[uttid] = scpid.split("_")[0]

    with open(text_raw, "r") as f:
        with open(text, "w") as ofh:
            for line in f:
                line = line.strip()
                segid = line.split()[0]
                segids.add(segid)
                # Skip adding spkid to the front if it's already there
                if len(segid.split()) == 7:
                    print(line, file=ofh)
                else:
                    uttid = segid2uttid(segid)
                    # Temporary hack for solving the issue of the change of mapping in experiments
                    if uttid in uttid2spkid:
                        print("_".join([uttid2spkid[uttid], line]), file=ofh)

    with open(utt2spk, "w") as fp:
        with open(output, "w") as ofh:
            with open(wav_scp, "r") as f:
                for line in f:
                    line = line.strip()
                    segid, segfp = line.split(maxsplit=1)
                    if segid not in segids:
                        continue
                    uttid = segid2uttid(segid)
                    if uttid in uttid2spkid:
                        spkid = uttid2spkid[uttid]
                        new_segfp = os.path.abspath(segfp)
                        # Skip adding spkid to the front if it's already there
                        if len(segid.split()) == 7:
                            print(f"{segid} {new_segfp}", file=ofh)
                            print(f"{segid} {spkid}", file=fp)
                        else:
                            # The spkid is added in the front of the old uttid
                            new_segid = "_".join([spkid, segid])
                            print(f"{new_segid} {new_segfp}", file=ofh)
                            print(f"{new_segid} {spkid}", file=fp)


def main():
    parser = argparse.ArgumentParser(
        description='Determine anchor utterances based on the aligned text file.')
    parser.add_argument('--scp_map', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    parser.add_argument('--wav_scp', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    args = parser.parse_args()

    filter_aligned_utt(args.scp_map, args.wav_scp, args.output_dir)


if __name__ == "__main__":
    main()
