#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=zh-HK # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk

# train_set=train_"$(echo "${lang}" | tr - _)"
# train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set="decode"
# target_dir="/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/vad_align_zh-HK/segments/M17020002_23_03-35-42_03-45-25_胡志偉議員/clips"

asr_config=conf/tuning/train_asr_conformer5.yaml
lm_config=conf/train_lm.yaml
lm_config_small=conf/train_lm_small.yaml
inference_config=conf/decode_ngram.yaml
asr_model_dir=exp/asr_train_asr_conformer5_raw_zh-HK_word_sp
lm_exp=exp/lm_train_lm_zh-HK_word
lm_tag=special

if [[ "${lang}" == *"zh"* ]]; then
  nbpe=2500
elif [[ "${lang}" == *"fr"* ]]; then
  nbpe=350
elif [[ "${lang}" == *"es"* ]]; then
  nbpe=235
else
  nbpe=150
fi

./decode_vad.sh \
    --ngpu 2 \
    --lang "${lang}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type word \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --test_sets ${test_set} \
    --asr_tag  "decode_vad" \
    --asr_model_dir ${asr_model_dir} \
    --lm_tag ${lm_tag} \
    --inference_nj 2 \
    --phoneme_align true \
    --stage 13 \
    --stop_stage 14

# ./decode_vad.sh \
#     --ngpu 4 \
#     --lang "${lang}" \
#     --use_lm true \
#     --lm_config "${lm_config}" \
#     --token_type word \
#     --nbpe $nbpe \
#     --feats_type raw \
#     --speed_perturb_factors "0.9 1.0 1.1" \
#     --asr_config "${asr_config}" \
#     --inference_config "${inference_config}" \
#     --test_sets ${test_set} \
#     --asr_tag  "decode_vad" \
#     --asr_model_dir ${asr_model_dir} \
#     --lm_exp ${lm_exp} \
#     --inference_nj 2 \
#     --skip_lm_train true \
#     --use_special_lm false \
#     --phoneme_align true \
#     --stage 13 \
#     --stop_stage 13