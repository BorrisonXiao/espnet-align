#!/usr/bin/env python3

from utils import read_anchor_file
import os

PATH = "/home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/align_exp_v4/align_decode_vad_3gram_biased/decode_k2_ngram_ngram_3gram_asr_model_valid.acc.best_use_k2_k2_ctc_decoding_true_use_nbest_rescoring_true/decode/anchors/raw/char"

def check_anchors():
    for file in os.listdir(PATH):
        if file.endswith(".anchor"):
            try:
                fname, uttid, ref, hyp, op, csid = read_anchor_file(os.path.join(PATH, file), return_uttid=True)
            except:
                print(file)

check_anchors()