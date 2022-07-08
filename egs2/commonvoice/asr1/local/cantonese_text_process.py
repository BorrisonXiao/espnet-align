'''
Cantonese text process:
1. remove punctuations
2. (Optional) word segmentation (based on Jieba)
'''

import jieba
import sys
import argparse
import re

chinese_punc = re.compile(r'\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?')


def clean_text(input_filename, output_filename, word_seg=True):
    ofh = open(output_filename, 'w', encoding='utf-8')
    with open(input_filename, 'r', encoding='utf-8') as fhd:
        for line in fhd:
            line = re.subn(chinese_punc, '', line.strip())[0]
            if line:
                if word_seg:
                    line = ' '.join(jieba.cut(line, cut_all=False))
                print(line, file=ofh)
    ofh.close()


def clean_cv_text(input_filename, output_filename, word_seg=True):
    ofh = open(output_filename, 'w', encoding='utf-8')
    with open(input_filename, 'r', encoding='utf-8') as fhd:
        for line in fhd:
            uttid, text = line.split(sep=" ", maxsplit=1)
            if text:
                if word_seg:
                    text = ' '.join(jieba.cut(text, cut_all=False))

                res = " ".join(
                    [uttid, re.subn(chinese_punc, '', text.strip())[0]])
                print(res, file=ofh)
    ofh.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input: raw, plain text file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output: processed, plain text file')
    parser.add_argument('--word_seg', action='store_true',
                        help='If option provided, perform the word segmentaion (for Cantonese)')
    parser.add_argument('--commonvoice', action='store_true',
                        help='If option provided, perform text process on the commonvoice dataset')
    args = parser.parse_args()

    if args.commonvoice:
        clean_cv_text(args.input, args.output, word_seg=args.word_seg)
    else:
        clean_text(args.input, args.output, word_seg=args.word_seg)


if __name__ == "__main__":
    main()
