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
from utils import mkdir_if_not_exist

chinese_punc = re.compile(r'\!|\;|\~|\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?|\'|\"')


def clean_hklegco_text(input_dir, output_dir, use_jieba=False):
    """
    Re-tokenize the raw script files and generate a text_map file.
    The text_map file follows the following format:
        <uttid> <path_to_scp>
    where each scp file consists of one line for each utterance.
    """
    output_dir = os.path.abspath(output_dir)
    text_map = os.path.join(output_dir, "text_map")
    sent_text_map = os.path.join(output_dir, "sent_text_map")
    dump_dir = os.path.join(output_dir, "dump")
    sent_dump = os.path.join(output_dir, "sent_dump")
    mkdir_if_not_exist(dump_dir)
    mkdir_if_not_exist(sent_dump)

    with open(sent_text_map, "w") as sent_tm:
        with open(text_map, "w") as ofh:
            for mid in os.listdir(input_dir):
                mid_dir = os.path.join(input_dir, mid, "can")
                for txt_f in os.listdir(mid_dir):
                    uttid = Path(txt_f).stem
                    full_path = os.path.join(mid_dir, txt_f)

                    # TODO: Parallelize the re-tokenization process if needed
                    with open(full_path, 'r', encoding='utf-8') as fhd:
                        output_path = os.path.join(dump_dir, uttid + ".txt")
                        sent_output = os.path.join(sent_dump, uttid + ".txt")
                        with open(output_path, "w") as f:
                            with open(sent_output, "w") as sent_ofh:
                                for i, line in enumerate(fhd):
                                    if line:
                                        splitted = line.strip().split(maxsplit=1)
                                        if len(splitted) <= 1:
                                            continue
                                        sentid, text = splitted
                                        # No longer needed as this is done in the updated preprocess pipline
                                        # text = re.subn(chinese_punc, '', text)[0]
                                        if use_jieba:
                                            text = ' '.join(
                                                jieba.cut(text, cut_all=False))
                                            res = " " + text if i > 0 else text
                                            print(text, file=sent_ofh)
                                        else:
                                            # No longer needed as this is done in the updated preprocess pipline
                                            # processed_text = text_char_seg(text)
                                            processed_text = text
                                            # print(sentid, processed_text)
                                            res = " " + \
                                                processed_text if i > 0 else processed_text
                                            # Keep sentence-level segmentation info
                                            print(sentid, processed_text, file=sent_ofh)

                                        print(res, file=f, end="")

                    print(f"{uttid} {output_path}", file=ofh)
                    print(f"{uttid} {sent_output}", file=sent_tm)


def clean_cv_text(input_filename, output_filename, use_jieba=False):
    ofh = open(output_filename, 'w', encoding='utf-8')
    with open(input_filename, 'r', encoding='utf-8') as fhd:
        for line in fhd:
            uttid, text = line.split(sep=" ", maxsplit=1)
            if text:
                if use_jieba:
                    text = ' '.join(jieba.cut(text, cut_all=False))

                text = re.subn(chinese_punc, '', text.strip())[0]
                res = text_char_seg(text)
                print(f"{uttid} {res}", file=ofh)
    ofh.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='The full path to the directory in which original text files are stored')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which tokenized text files are stored and a text_map/sent_text_map file')
    parser.add_argument('--use_jieba', action='store_true',
                        help='If option provided, use jieba to segment the text else space-delimited tokenization')
    parser.add_argument('--commonvoice', action='store_true',
                        help='If option provided, perform text process on the commonvoice dataset')
    args = parser.parse_args()

    if args.commonvoice:
        clean_cv_text(args.input_dir, args.output_dir, use_jieba=args.use_jieba)
    else:
        clean_hklegco_text(args.input_dir, args.output_dir, use_jieba=args.use_jieba)


if __name__ == "__main__":
    main()
