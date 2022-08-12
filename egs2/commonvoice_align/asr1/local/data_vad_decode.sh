#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Cihan Xiao)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1
stop_stage=2
SECONDS=0
lang=zh-HK
python=python3
local_data_dir=
txt_data_dir=
dst=data/decode
mkdir -p $dst

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing post-vad hklegco data"
    
    ${python} local/decode_data_prep.py --data_dir ${local_data_dir}/decode --output $dst/wav.scp.unsorted --utt2spk_output $dst/utt2spk.unsorted
    sort -o $dst/utt2spk $dst/utt2spk.unsorted
    rm $dst/utt2spk.unsorted
    sort -o $dst/wav.scp $dst/wav.scp.unsorted
    rm $dst/wav.scp.unsorted

    spk2utt=$dst/spk2utt
    utils/utt2spk_to_spk2utt.pl <$dst/utt2spk >$spk2utt || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
    log "stage2: Tokenize text data using char/jieba and yield a text file"
    ${python} local/cantonese_text_process.py --input_dir ${txt_data_dir} --output_dir ${dst}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"