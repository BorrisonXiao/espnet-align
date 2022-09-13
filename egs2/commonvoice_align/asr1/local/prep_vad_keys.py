import argparse
from pathlib import Path
import os
import re

punc = re.compile(r'\(|\)|\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?')


def generate_keys(input_dir, output):
    with open(output, "w") as f:
        for mid in os.listdir(input_dir):
            mid_dir = os.path.join(input_dir, mid)
            for utt in os.listdir(mid_dir):
                uttid = re.subn(punc, '', Path(utt).stem.replace(" ", ""))[0]
                utt_fp = os.path.join(mid_dir, utt)
                print(f"{uttid} {utt_fp}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create key files for vad sementation.')
    parser.add_argument('--input_dir',
                        type=Path,
                        required=True,
                        help='The directory containing the input audio files.')
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='The output file in which the keys are stored.')
    args = parser.parse_args()

    generate_keys(args.input_dir, args.output)
