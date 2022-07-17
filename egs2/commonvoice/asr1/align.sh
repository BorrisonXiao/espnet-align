# ASR model and config files from pretrained model (e.g. from cachedir):
asr_config=/home/cxiao7/research/espnet/egs2/commonvoice/asr1/exp/asr_train_asr_conformer5_fbank_pitch_zh-HK_word_sp/config.yaml
asr_model=/home/cxiao7/research/espnet/egs2/commonvoice/asr1/exp/asr_train_asr_conformer5_fbank_pitch_zh-HK_word_sp/valid.acc.best.pth

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
raw_data_dir="/home/cxiao7/research/speech2text/test_data/video/M17020002/can/clips"
text_dir="/home/cxiao7/research/speech2text/test_data/txt/2017-02-08/can/word_seg"
audio_data_dir="$(dirname "${raw_data_dir}")/data"
txt_data_dir="$(dirname "${text_dir}")/data"
stage=4
stop_stage=4
nj=32
jobname="align.sh"
ngpu=1
num_nodes=1
align_exp="./exp/align_zh-HK"
python=python3
clip_audio=true
clip_output_dir="${align_exp}/clips"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for audio in ${raw_data_dir}."
    mkdir -p $audio_data_dir
    ${python} ./local/prepare_hklegco_wav.py --input $raw_data_dir --output $audio_data_dir/wav.scp

    # Feature extraction to match fbank_pitch
    _nj=$(min "${nj}" "$(<"${audio_data_dir}/wav.scp" wc -l)")
    steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${audio_data_dir}"
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
    text="/home/cxiao7/research/speech2text/test_data/txt/2017-02-08/can/data"
    wav="/home/cxiao7/research/speech2text/test_data/video/M17020002/can/data"
    mkdir -p $align_exp; mkdir -p "${align_exp}"/segments
    _nj=1
    ${cuda_cmd} --gpu "${ngpu}" JOB=1:"${_nj}" "${align_exp}"/align.JOB.log \
        ${python} ./local/asr_align_cv.py \
            --asr_train_config ${asr_config} \
            --asr_model_file ${asr_model} \
            --data_path_and_name_and_type "${wav}/feats.scp,speech,kaldi_ark" \
            --text_dir ${text} \
            --metadata_dir ${wav} \
            --output_dir "${align_exp}"/segments \
            --min_window_size 1600 \
            --gratis_blank true\
            --log_level "DEBUG"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ ${clip_audio} ]; then
    log "Stage 4: Clip audio files based on the segment files."
    mkdir -p $clip_output_dir

    # Requires installation of pydub and ffmpeg
    # e.g. pip install ffmpeg; pip install pydub
    ${python} ./local/clip_segments.py \
        --seg_file_dir "${align_exp}"/segments \
        --audio_dir $raw_data_dir \
        --output_dir $clip_output_dir
fi