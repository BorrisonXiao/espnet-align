# ASR model and config files from pretrained model (e.g. from cachedir):
asr_config=/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/asr_train_asr_transformer3_w2v_large_lv60_960h_finetuning_last_1layer_raw_zh-HK_word_sp_prev/config.yaml
asr_model=/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/asr_train_asr_transformer3_w2v_large_lv60_960h_finetuning_last_1layer_raw_zh-HK_word_sp_prev/valid.acc.ave_10best.pth

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

. ./path.sh
. ./cmd.sh

# Prepare the text file
align_exp="./exp/align_zh-HK"
raw_data_dir="/home/cxiao7/research/speech2text/test_data/video/M17020002/can/clips"
text_dir="/home/cxiao7/research/speech2text/test_data/txt/2017-02-08/can/word_seg"
audio_data_dir="${align_exp}/data/audio"
txt_data_dir="${align_exp}/data/txt"
stage=3
stop_stage=3
nj=32
jobname="align.sh"
ngpu=1
num_nodes=1
python=python3
clip_audio=true
clip_output_dir="${align_exp}/clips"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for audio in ${raw_data_dir}."
    mkdir -p $audio_data_dir
    ${python} ./local/prepare_hklegco_wav.py --input_dir $raw_data_dir --output_dir $audio_data_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Data preparation for text in ${text_dir}."
    mkdir -p $txt_data_dir

    for file in $text_dir/*.txt; do
        filename="${file##*/}"
        # Process the text to keep the asr_align.py script happy (<uttid> <segment>)
        ${python} ./local/prepare_txt_align.py --input $file --output "${txt_data_dir}/${filename}"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Obtain alignments."
    # TODO: txt2audio mapping and batch segmentation
    text="/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/align_zh-HK/data/txt/M17020002_23_03-35-42_03-45-25_胡志偉議員.txt"
    wav="/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/align_zh-HK/data/audio/M17020002_23_03-35-42_03-45-25_胡志偉議員.wav"
    mkdir -p $align_exp; mkdir -p "${align_exp}"/segments
    _nj=1
    ${cuda_cmd} --gpu "${ngpu}" JOB=1:"${_nj}" "${align_exp}"/align.JOB.log \
        ${python} ./local/asr_align_hklegco.py \
            --asr_train_config ${asr_config} \
            --asr_model_file ${asr_model} \
            --audio ${wav} \
            --text ${text} \
            --output "${align_exp}"/segments/"M17020002_23_03-35-42_03-45-25_胡志偉議員.seg.txt" \
            --min_window_size 1024000
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ ${clip_audio} ]; then
    log "Stage 4: Clip audio files based on the segment files."
    mkdir -p $clip_output_dir

    # Requires installation of pydub and ffmpeg
    # e.g. pip install ffmpeg; pip install pydub
    # ${python} ./local/clip_segments.py \
    #     --seg_file_dir "${align_exp}"/segments \
    #     --audio_dir $raw_data_dir \
    #     --output_dir $clip_output_dir
    
    ${python} ./local/clip_segments.py \
        --seg_file_dir "/home/cxiao7/research/speech2text/processed_data/txt" \
        --audio_dir $raw_data_dir \
        --output_dir "/home/cxiao7/research/speech2text/processed_data/audio/clips"
fi