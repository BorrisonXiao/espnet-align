import argparse
import os
from pathlib import Path


def generate_scp(input_dir, output_filename, utt2spk_output, segments_output):
    """
    All audio files are merged into one single giant wav.scp for efficient decoding.
    """
    ofh = open(output_filename, 'w', encoding='utf-8')
    utt2spk = open(utt2spk_output, 'w', encoding='utf-8')
    segments = open(segments_output, "w", encoding='utf-8')
    for uttid in os.listdir(input_dir):
        spk = uttid.strip().split("_")[0]
        if segments_output:
            # Merge the "wav.scp" files
            wav_scp_fp = os.path.join(input_dir, uttid, "wav.scp")
            with open(wav_scp_fp, "r") as f:
                splits = f.read().rstrip().split(" ")
                scp_uttid, full_path = splits
                assert scp_uttid == uttid, "The uttid in {wav_scp_fp} does not match with its uttid in its dirname."
                print(
                    f"{uttid} ffmpeg -i {full_path} -f wav -ar 16000 -ab 16 -ac 1 - |", file=ofh)

            # Merge the "segments" files
            segments_fp = os.path.join(input_dir, uttid, "segments")
            with open(segments_fp, "r") as f:
                for line in f:
                    line = line.rstrip()
                    segid, uttid, _, _ = line.split(" ")
                    print(line, file=segments)
                    print(f"{segid} {spk}", file=utt2spk)
        else:
            clips_dir = os.path.join(input_dir, uttid, "clips")
            for clip_fname in os.listdir(clips_dir):
                segid = Path(clip_fname).stem
                full_path = os.path.join(clips_dir, clip_fname)
                print(
                    f"{segid} ffmpeg -i {full_path} -f wav -ar 16000 -ab 16 -ac 1 - |", file=ofh)
                print(f"{segid} {spk}", file=utt2spk)
    ofh.close()
    utt2spk.close()
    segments.close()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the HKLEGCO audio data')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Input: The directory containing the original audio files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output: The generated wav.scp file.')
    parser.add_argument('--utt2spk_output', type=Path, required=True,
                        help='Output: The generated utt2spk file.')
    parser.add_argument('--segments_output', type=Path, default=None,
                        help='Output: The generated utt2spk file.')
    args = parser.parse_args()

    generate_scp(args.data_dir, args.output, args.utt2spk_output, args.segments_output)


if __name__ == "__main__":
    main()
