#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lm_train_text=
ngram_dir=
ngram_num=4

. utils/parse_options.sh

. ./path.sh

cut -f 2- -d " " "${lm_train_text}" | lmplz -S "20%" --discount_fallback -o ${ngram_num} - >"${ngram_dir}"/${ngram_num}gram.arpa
build_binary -s "${ngram_dir}"/${ngram_num}gram.arpa "${ngram_dir}"/${ngram_num}gram.bin
