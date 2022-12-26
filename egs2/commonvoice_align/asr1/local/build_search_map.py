import argparse
import os
from pathlib import Path


def build_map(input_dir, output, use_phoneme=True):
    token_type = "phoneme" if use_phoneme else "char"
    with open(output, "w") as f:
        for mid in os.listdir(input_dir):
            if mid != "M19120002":
                continue
            hyp = os.path.join(input_dir, mid, "hyp", token_type, mid + ".hyp")
            ref = os.path.join(input_dir, mid, "ref", token_type, mid + ".ref")
            assert os.path.exists(hyp) and os.path.exists(ref), f"{hyp}, {ref}"
            print(hyp, ref, file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Build search map.')
    parser.add_argument('--dir', type=Path, required=True,
                        help='The full path to decoded directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to output search map file')
    parser.add_argument('--use_phoneme', action='store_true',
                        help='If option provided, the text will be converted to phoneme using pinyin_jyutping_sentence')
    parser.add_argument('--ignore_tone', action='store_true',
                        help='If option provided, the converted phonemes will disregard tones')
    args = parser.parse_args()

    build_map(input_dir=args.dir, output=args.output, use_phoneme=args.use_phoneme)


if __name__ == "__main__":
    main()
