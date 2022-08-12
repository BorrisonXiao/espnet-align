import argparse
import os
from pathlib import Path


def generate_scp(input_dir, output_filename, utt2spk_output):
    """
    All audio files are merged into one single giant wav.scp for efficient decoding.
    """
    ofh = open(output_filename, 'w', encoding='utf-8')
    utt2spk = open(utt2spk_output, 'w', encoding='utf-8')
    for uttid in os.listdir(input_dir):
        spk = uttid.strip().split("_")[0]
        clips_dir = os.path.join(input_dir, uttid, "clips")
        for clip_fname in os.listdir(clips_dir):
            segid = Path(clip_fname).stem
            full_path = os.path.join(clips_dir, clip_fname)
            print(f"{segid} ffmpeg -i {full_path} -f wav -ar 16000 -ab 16 -ac 1 - |", file=ofh)
            print(f"{segid} {spk}", file=utt2spk)
    ofh.close()
    utt2spk.close()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the HKLEGCO audio data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Input: The directory containing the original .mp3 files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output: The generated wav.scp file.')
    parser.add_argument('--utt2spk_output', type=str, required=True,
                        help='Output: The generated utt2spk file.')
    args = parser.parse_args()

    generate_scp(args.data_dir, args.output, args.utt2spk_output)


if __name__ == "__main__":
    main()
