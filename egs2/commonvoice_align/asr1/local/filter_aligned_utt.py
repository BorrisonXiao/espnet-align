import argparse
from pathlib import Path


def filter_aligned_utt(scp_map, wav_scp, output, utt2spk):
    """
    Filter the wav.scp file from based on the primary alignment result to form new dataset.
    """
    aligned_audio = {}  # Stores the {uttid: spkid} pair
    with open(scp_map, "r") as f:
        for line in f:
            uttid, scpid = line.strip().split(maxsplit=1)
            # The spkid by default is the first part of the scpid
            aligned_audio[uttid] = scpid.split("_")[0]

    with open(utt2spk, "w") as fp:
        with open(output, "w") as ofh:
            with open(wav_scp, "r") as f:
                for line in f:
                    line = line.strip()
                    uttid = line.split()[0]
                    if uttid in aligned_audio:
                        print(line, file=ofh)
                        print(f"{uttid} {aligned_audio[uttid]}", file=fp)


def main():
    parser = argparse.ArgumentParser(
        description='Determine anchor utterances based on the aligned text file.')
    parser.add_argument('--scp_map', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    parser.add_argument('--wav_scp', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    parser.add_argument('--utt2spk', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    args = parser.parse_args()

    filter_aligned_utt(args.scp_map, args.wav_scp, args.output, args.utt2spk)


if __name__ == "__main__":
    main()
