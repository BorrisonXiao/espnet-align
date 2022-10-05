import soundfile as sf
import argparse
from pathlib import Path


def calc_audio_len(wav_scp, tag):
    """
    Check the length of all audio files in a dataset.
    """
    audio_len = 0
    audio_num = 1e-12
    with open(wav_scp, "r") as f:
        for line in f:
            _, wavfp = line.strip().split(maxsplit=1)
            wavf = sf.SoundFile(wavfp)
            audio_len += float(wavf.frames / wavf.samplerate)
            audio_num += 1
    if tag:
        print(
            f"The length of the {tag} set is {audio_len:.2f} seconds, with {audio_len / audio_num:.2f} seconds per segment on average.")
    else:
        print(
            f"The length of the set is {audio_len:.2f} seconds, with {audio_len / audio_num:.2f} seconds per segment on average.")
    return audio_len, audio_num


def main():
    parser = argparse.ArgumentParser(
        description='Calculate the audio length of a given dataset (in .wav).')
    parser.add_argument('--wav_scp', type=Path, required=True,
                        help='The wav.scp file of the target dataset.')
    parser.add_argument('--tag', type=str, default=None,
                        help='The tag of the target dataset.')
    args = parser.parse_args()

    calc_audio_len(args.wav_scp, args.tag)


if __name__ == "__main__":
    main()
