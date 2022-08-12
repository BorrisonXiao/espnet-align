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
align_exp="./exp/vad_align_zh-HK"
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

audio_dir="/home/cxiao7/research/espnet-cxiao/egs2/hklegco/asr1/exp/align_zh-HK/data/audio"
mkdir -p $align_exp; mkdir -p $align_exp/segments

${python} local/vad.py --input_dir $audio_dir --output_dir $align_exp/segments --fs 16000 --clip