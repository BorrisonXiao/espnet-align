#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /home/cxiao7/research/speech2text/processed_data/dev data/dev"
  exit 1
fi

src=$1
dst=$2
audio_dir=$src/audio
txt_dir=$src/txt
stage=1
stop_stage=3
python=python3

# spk_file=$src/../SPEAKERS.TXT

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1
# [ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

# wav.scp file generation
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  find -L $audio_dir/ -iname "*.wav" | sort | xargs -I% basename % .wav | \
    awk -v "dir=$audio_dir" '{printf "%s ffmpeg -i %s/%s.wav -f wav -ar 16000 -ab 16 -ac 1 - |\n", $0, dir, $0}' >> $wav_scp || exit 1
fi

# text and utt2spk file generation
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ${python} ./local/seg2text.py --seg_file_dir $txt_dir --text_output $dst/text --utt2spk_output $dst/utt2spk --char_seg
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  spk2utt=$dst/spk2utt
  utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

  ntrans=$(wc -l <$trans)
  nutt2spk=$(wc -l <$utt2spk)
  ! [ "$ntrans" -eq "$nutt2spk" ] && \
    echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

  utils/validate_data_dir.sh --no-feats $dst || exit 1
fi

echo "$0: successfully prepared data in $dst"

exit 0
