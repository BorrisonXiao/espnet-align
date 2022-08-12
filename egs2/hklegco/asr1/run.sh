#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev"

asr_config=conf/tuning/train_asr_transformer3_w2v_large_lv60_960h_finetuning_last_1layer.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang zh-HK \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 1000 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --feats_normalize utt_mvn \
    --stage 12 \
    --stop-stage 12
