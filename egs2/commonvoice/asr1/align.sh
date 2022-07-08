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
text_dir="/home/cxiao7/research/speech2text/test_data/txt/2017-02-08/can/sent_seg"
audio_data_dir="$(dirname "${raw_data_dir}")/data"
txt_data_dir="$(dirname "${text_dir}")/data"
stage=2
stop_stage=2
nj=32

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for audio in ${raw_data_dir}."
    mkdir -p $audio_data_dir
    python ./local/prepare_hklegco_wav.py --input $raw_data_dir --output $audio_data_dir/wav.scp

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
        python ./local/prepare_txt_align.py --input $file --output "${txt_data_dir}/${filename}"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Obtain alignments."
    # TODO: txt2audio mapping
    text="/home/cxiao7/research/speech2text/test_data/txt/2017-02-08/can/data/2017-02-08_huzhiwei.txt"
    wav=
    python $MAIN_ROOT/espnet2/bin/asr_align.py --asr_train_config ${asr_config} --asr_model_file ${asr_model} --audio ${wav} --text ${text}
fi