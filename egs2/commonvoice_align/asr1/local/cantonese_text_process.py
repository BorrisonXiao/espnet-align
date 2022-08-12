'''
Cantonese text process:
1. remove punctuations
2. (Optional) word segmentation (based on Jieba)
'''

import jieba
import argparse
from seg2text import text_char_seg
import re
import os
from pathlib import Path

chinese_punc = re.compile(r'\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?')


def mkdir_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_hklegco_text(input_dir, output_dir, use_jieba=False):
    """
    Re-tokenize the raw script files and generate a text_map file.
    The text_map file follows the following format:
        <uttid> <path_to_scp>
    where each scp file consists of one line for each utterance.
    """
    output_dir = os.path.abspath(output_dir)
    text_map = os.path.join(output_dir, "text_map")
    dump_dir = os.path.join(output_dir, "dump")
    mkdir_if_not_exist(dump_dir)

    with open(text_map, "w") as ofh:
        for txt_f in os.listdir(input_dir):
            uttid = Path(txt_f).stem
            full_path = os.path.join(input_dir, txt_f)

            # TODO: Parallelize the re-tokenization process if needed
            with open(full_path, 'r', encoding='utf-8') as fhd:
                output_path = os.path.join(dump_dir, uttid + ".txt")
                with open(output_path, "w") as f:
                    for i, line in enumerate(fhd):
                        if line:
                            text = re.subn(chinese_punc, '', line.strip())[0]
                            if use_jieba:
                                text = ' '.join(jieba.cut(text, cut_all=False))
                                res = " " + text if i > 0 else text
                            else:
                                res = " " + \
                                    text_char_seg(
                                        text) if i > 0 else text_char_seg(text)

                            print(res, file=f, end="")

            print(f"{uttid} {output_path}", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='The full path to the directory in which original text files are stored')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which tokenized text files are stored and a text_map file')
    parser.add_argument('--use_jieba', action='store_true',
                        help='If option provided, use jieba to segment the text else space-delimited tokenization')
    args = parser.parse_args()

    clean_hklegco_text(args.input_dir, args.output_dir, use_jieba=args.use_jieba)


if __name__ == "__main__":
    main()
