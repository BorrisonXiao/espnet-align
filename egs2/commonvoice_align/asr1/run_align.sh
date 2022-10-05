#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=zh-HK
test_set="decode"

asr_config=conf/tuning/train_asr_conformer5.yaml
asr_lm_config=conf/train_lm.yaml
inference_config=conf/decode_ngram.yaml
asr_inference_config=conf/decode_asr.yaml
asr_model_dir=pre_train_exp/asr_train_asr_conformer5_raw_zh-HK_word_sp
finetune_asr_config=conf/tuning/finetune_asr_conformer5.yaml
lm_tag=biased

input_text_dir="/home/cxiao7/research/speech2text/align_data_v1/processed/txt"

if [[ "${lang}" == *"zh"* ]]; then
  nbpe=2500
elif [[ "${lang}" == *"fr"* ]]; then
  nbpe=350
elif [[ "${lang}" == *"es"* ]]; then
  nbpe=235
else
  nbpe=150
fi

# Command for training ASR
./align.sh \
  --ngpu 2 \
  --nj 64 \
  --lang "${lang}" \
  --use_lm false \
  --token_type word \
  --nbpe $nbpe \
  --feats_type raw \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --test_sets ${test_set} \
  --asr_tag "decode_vad" \
  --asr_exp ${asr_model_dir} \
  --expdir align_exp \
  --lm_tag ${lm_tag} \
  --inference_nj 64 \
  --phoneme_align true \
  --pretrain_asr true \
  --asr_lm_config ${asr_lm_config} \
  --asr_inference_config ${asr_inference_config} \
  --heuristic_search false \
  --finetune_asr_config ${finetune_asr_config} \
  --input_text_dir ${input_text_dir} \
  --compute_primary_stats true \
  --stage 17 \
  --stop_stage 17

# ./align.sh \
#     --ngpu 2 \
#     --lang "${lang}" \
#     --use_lm false \
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
#     --lm_tag ${lm_tag} \
#     --inference_nj 2 \
#     --phoneme_align true \
#     --stage 13 \
#     --stop_stage 14

# ./align.sh \
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
