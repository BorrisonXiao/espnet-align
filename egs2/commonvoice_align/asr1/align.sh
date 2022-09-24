#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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
SECONDS=0

# General
vad_data_dir=data/vad
input_text_dir="/home/cxiao7/research/speech2text/align_data_v0/processed/txt"
input_audio_dir="/home/cxiao7/research/speech2text/align_data_v0/processed/audio"
skip_lm_train=false
seg_file_format="kaldi" # Segment file format: "json", "kaldi"
pretrain_asr=false
vad_nj=16
vad_mthread=2

# Bootstrapping realted
max_iter=1

# Pre-train related
asr_inference_config=
asr_lm_config=

# Alignment-related
use_biased_lm=true
phoneme_align=false
ignore_tone=true
eps="***"
align_exp=
heuristic_search=true
wrap_primary_results=true
wrap_primary_results_norm=true

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
align_dir=

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=wav     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=word                   # Tokenization type (char or bpe).
nbpe=30                           # The number of BPE vocabulary.
bpemode=unigram                   # Mode of BPE (unigram or bpe).
oov="<unk>"                       # Out of vocabulary symbol.
blank="<blank>"                   # CTC blank symbol
sos_eos="<sos/eos>"               # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=                       # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0                # character coverage when modeling BPE

# Ngram model related
use_ngram=true
ngram_exp=
ngram_num=3

# Language model related
use_lm=true # Use language model for ASR decoding.
lm_tag=     # Suffix to the result dir for language model training.
lm_exp=     # Specify the directory path for LM experiment.
# If this option is specified, lm_tag is ignored.
lm_stats_dir= # Specify the directory path for LM statistics.
lm_config=    # Config for language model training.
lm_args=      # Arguments for language model training, e.g., "--max_epoch 10".
# Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag= # Suffix to the result dir for asr model training.
asr_exp= # Specify the directory path for ASR experiment.
# If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the directory path for ASR statistics.
asr_config=    # Config for asr model training.
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
# Note that it will overwrite args in asr config.
pretrained_model=          # Pretrained model to load
ignore_init_mismatch=false # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.
# Decoding-only
asr_model_dir=

# Upload model related
hf_repo=

# Decoding related
use_k2=false # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
# and transformer language model for nbest rescoring
num_paths=1000     # The 3rd argument of k2.random_paths.
nll_batch_size=100 # Affect GPU memory usage when computing nll
# during nbest rescoring
k2_config=./conf/decode_asr_transformer_with_k2.yaml

use_streaming=false # Whether to use streaming decoding

use_maskctc=false # Whether to use maskctc decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
# Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
# e.g.
# inference_asr_model=train.loss.best.pth
# inference_asr_model=3epoch.pth
# inference_asr_model=valid.acc.best.pth
# inference_asr_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=                 # Name of training set.
valid_set=                 # Name of validation set used for monitoring/tuning network training.
test_sets=                 # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=            # Text file path of bpe training set.
lm_train_text=             # Text file path of language model training set.
lm_dev_text=               # Text file path of language model development set.
lm_test_text=              # Text file path of language model evaluation set.
nlsyms_txt=none            # Non-linguistic symbol list if existing.
cleaner=none               # Text cleaner.
g2p=none                   # g2p method (needed if token_type=phn).
lang=noinfo                # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag          # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_exp          # Specify the directory path for ASR experiment.
                       # If this option is specified, asr_tag is ignored (default="${asr_exp}").
    --asr_stats_dir    # Specify the directory path for ASR statistics (default="${asr_stats_dir}").
    --asr_config       # Config for asr model training (default="${asr_config}").
    --asr_args         # Arguments for asr model training (default="${asr_args}").
                       # e.g., --asr_args "--max_epoch 10"
                       # Note that it will overwrite args in asr config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_asr_model # ASR model path for decoding (default="${inference_asr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --use_streaming       # Whether to use streaming decoding (default="${use_streaming}").
    --use_maskctc         # Whether to use maskctc decoding (default="${use_streaming}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --asr_speech_fold_length # fold_length for speech data during ASR training (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check required arguments
# [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
# [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && {
    log "${help_message}"
    log "Error: --test_sets is required"
    exit 2
}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi

# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        asr_tag+="_${lang}_${token_type}"
    else
        asr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi
if [ -z "${align_dir}" ]; then
    align_dir=${expdir}/align_${asr_tag}_${lm_tag}
fi

# The directory used for collect-stats mode
if [ -z "${asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${lang}_${token_type}"
    else
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${asr_exp}" ]; then
    asr_exp="${expdir}/asr_${asr_tag}"
fi
if [ -z "${align_exp}" ]; then
    align_exp="${expdir}/align_${asr_tag}"
    if "${use_lm}"; then
        align_exp+="_${lm_config}"
    fi
    if "${use_ngram}"; then
        align_exp+="_${ngram_num}gram"
    fi
    if "${use_biased_lm}"; then
        align_exp+="_biased"
    else
        align_exp="_general"
    fi
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
        inference_tag+="_use_k2"
        inference_tag+="_k2_ctc_decoding_${k2_ctc_decoding}"
        inference_tag+="_use_nbest_rescoring_${use_nbest_rescoring}"
    fi
fi

# ========================== Main stages start from here. ==========================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Run VAD on the original data."
    mkdir -p ${vad_data_dir}/decode
    _logdir=${vad_data_dir}/dump/logdir
    mkdir -p ${_logdir}
    key_file=${vad_data_dir}/dump/vad_keys

    ${python} local/prep_vad_keys.py \
        --input_dir ${input_audio_dir} \
        --output ${key_file}

    _nj=$(min "${vad_nj}" "$(wc <${key_file} -l)")
    for n in $(seq "${_nj}"); do
        split_vads+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_vads}

    ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/align_vad.JOB.log \
        ${python} local/vad.py \
        --keyfile ${_logdir}/keys.JOB.scp \
        --output_dir ${vad_data_dir}/decode \
        --fs 16000 \
        --mthread ${vad_mthread} \
        --kaldi_output ||
        {
            cat $(grep -l -i error "${_logdir}"/align_vad.*.log)
            exit 1
        }
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Data preparation for decoding."
    # [Task dependent] Need to create data.sh for new corpus
    local/data_vad_decode.sh \
        --lang ${lang} \
        --local_data_dir ${vad_data_dir} \
        --txt_data_dir ${input_text_dir} \
        --dst data/decode \
        --kaldi_style true
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "${feats_type}" = raw ]; then
        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

            echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
        done
    fi
fi

if ! "${skip_lm_train}" && "${use_biased_lm}" || "${pretrain_asr}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Generate LM training text for each utterance"
        mkdir -p data/lm_train/utts
        rm -r data/lm_train/utts/*
        for dset in ${test_sets}; do
            # Generate lm_train.txt for each utterance
            ${python} local/generate_lm_train_text.py \
                --text_map data/${dset}/text_map \
                --output_dir data/lm_train/utts
        done
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 5: Generate merged lm_train.txt for further tokenization if needed"

            # TODO: Merge target reference text as well as the pre-train text to generate token list
            ${python} local/merge_lm_train.py \
                --utts_dir data/lm_train/utts \
                --output data/lm_train/lm_train.txt
        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi
    fi
fi

if "${pretrain_asr}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Train an ASR model using commonvoice-zh-HK data"
        asr_train_set=train_"$(echo "${lang}" | tr - _)"
        asr_train_dev=dev_"$(echo "${lang}" | tr - _)"
        asr_test_set="${asr_train_dev} test_$(echo ${lang} | tr - _)"

        ./pretrain_asr.sh \
            --ngpu "${ngpu}" \
            --lang "${lang}" \
            --use_lm true \
            --lm_config "${asr_lm_config}" \
            --token_type word \
            --feats_type raw \
            --speed_perturb_factors "0.9 1.0 1.1" \
            --asr_config "${asr_config}" \
            --asr_exp ${asr_exp} \
            --inference_config "${asr_inference_config}" \
            --train_set "${asr_train_set}" \
            --valid_set "${asr_train_dev}" \
            --test_sets "${asr_test_set}" \
            --lm_train_text "data/${asr_train_set}/text" "$@" \
            --expdir pre_train_exp \
            --add_token_text data/lm_train/lm_train.txt \
            --stage 11 \
            --stop_stage 11
    fi
fi

if ! "${skip_lm_train}" && "${use_biased_lm}" && [ ${stage} -le 10 ] && [ ${stop_stage} -ge 7 ]; then
    lm_stats_dir=${lm_stats_dir}_biased
    if "${use_lm}"; then
        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            log "Stage 7: Biased LM collect stats"

            # 1. Split the key file
            _logdir="${lm_stats_dir}/logdir"
            mkdir -p "${_logdir}"

            for dir in data/lm_train/*; do
                # Re-initialize _opts for each lm
                _opts=
                if [ -n "${lm_config}" ]; then
                    # To generate the config file: e.g.
                    #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                    _opts+="--config ${lm_config} "
                fi

                uttid=${dir##*/}
                key_file=data/lm_train/${uttid}/"lm_train.txt"
                mkdir -p ${_logdir}/${uttid}

                # Get the minimum number among ${nj} and the number lines of input files
                _nj=$(min "${nj}" "$(wc <${key_file} -l)")

                split_scps=""
                for n in $(seq ${_nj}); do
                    split_scps+=" ${_logdir}/${uttid}/train.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                # Using training set as dev set to make espnet happy
                split_scps=""
                for n in $(seq ${_nj}); do
                    split_scps+=" ${_logdir}/${uttid}/dev.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                # 2. Generate run.sh
                log "Generate '${lm_stats_dir}/${uttid}/run.sh'. You can resume the process from stage 7 using this script"
                mkdir -p "${lm_stats_dir}/${uttid}"
                echo "${run_args} --stage 7 \"\$@\"; exit \$?" >"${lm_stats_dir}/${uttid}/run.sh"
                chmod +x "${lm_stats_dir}/${uttid}/run.sh"

                # 3. Submit jobs
                log "LM collect-stats started... log: '${_logdir}/${uttid}/stats.*.log'"
                # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
                #       but it's used only for deciding the sample ids.
                # shellcheck disable=SC2046,SC2086
                ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/${uttid}/stats.JOB.log \
                    ${python} -m espnet2.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}" \
                    --token_list "${token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${key_file},text,text" \
                    --valid_data_path_and_name_and_type "${key_file},text,text" \
                    --train_shape_file "${_logdir}/${uttid}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/${uttid}/dev.JOB.scp" \
                    --output_dir "${_logdir}/${uttid}/stats.JOB" \
                    ${_opts} ${lm_args} || {
                    cat $(grep -l -i error "${_logdir}"/${uttid}/stats.*.log)
                    exit 1
                }

                # 4. Aggregate shape files
                _opts=
                for i in $(seq "${_nj}"); do
                    _opts+="--input_dir ${_logdir}/${uttid}/stats.${i} "
                done
                # shellcheck disable=SC2086
                ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}/${uttid}"

                # Append the num-tokens at the last dimensions. This is used for batch-bins count
                awk <"${lm_stats_dir}/${uttid}/train/text_shape" \
                    -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
                    >"${lm_stats_dir}/${uttid}/train/text_shape.${lm_token_type}"

                awk <"${lm_stats_dir}/${uttid}/valid/text_shape" \
                    -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
                    >"${lm_stats_dir}/${uttid}/valid/text_shape.${lm_token_type}"
            done
        fi

        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8-1: LM Training"

            for dir in data/lm_train/*; do
                _opts=
                if [ -n "${lm_config}" ]; then
                    # To generate the config file: e.g.
                    #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                    _opts+="--config ${lm_config} "
                fi
                uttid=${dir##*/}
                lm_trn_txt=data/lm_train/${uttid}/"lm_train.txt"

                if [ "${num_splits_lm}" -gt 1 ]; then
                    # If you met a memory error when parsing text files, this option may help you.
                    # The corpus is split into subsets and each subset is used for training one by one in order,
                    # so the memory footprint can be limited to the memory required for each dataset.

                    _split_dir="${lm_stats_dir}/${uttid}/splits${num_splits_lm}"
                    if [ ! -f "${_split_dir}/.done" ]; then
                        rm -f "${_split_dir}/.done"
                        ${python} -m espnet2.bin.split_scps \
                            --scps "${lm_trn_txt}" "${lm_stats_dir}/${uttid}/train/text_shape.${lm_token_type}" \
                            --num_splits "${num_splits_lm}" \
                            --output_dir "${_split_dir}"
                        touch "${_split_dir}/.done"
                    else
                        log "${_split_dir}/.done exists. Spliting is skipped"
                    fi

                    _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
                    _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
                    _opts+="--multiple_iterator true "

                else
                    _opts+="--train_data_path_and_name_and_type ${lm_trn_txt},text,text "
                    _opts+="--train_shape_file ${lm_stats_dir}/${uttid}/train/text_shape.${lm_token_type} "
                fi

                # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

                log "Generate '${lm_exp}/${uttid}/run.sh'. You can resume the process from stage 8 using this script"
                mkdir -p "${lm_exp}/${uttid}"
                echo "${run_args} --stage 8 \"\$@\"; exit \$?" >"${lm_exp}/${uttid}/run.sh"
                chmod +x "${lm_exp}/${uttid}/run.sh"

                log "LM training started... log: '${lm_exp}/${uttid}/train.log'"
                if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &>/dev/null; then
                    # SGE can't include "/" nor "zh-CN" in a job name
                    IFS='_' read -ra ADDR <<<${uttid}
                    # The job-id consists of the meeting ID and the metadata-based segment ID
                    jobname="$(basename ${lm_exp})_${ADDR[1]}_${ADDR[2]}"
                else
                    jobname="${lm_exp}/train.log"
                fi

                _ngpu=1 # Needs only 1 GPU due to the one-line text training data

                # shellcheck disable=SC2086
                ${python} -m espnet2.bin.launch \
                    --cmd "${cuda_cmd} --name ${jobname}" \
                    --log "${lm_exp}/${uttid}"/train.log \
                    --ngpu "${_ngpu}" \
                    --num_nodes "${num_nodes}" \
                    --init_file_prefix "${lm_exp}/${uttid}"/.dist_init_ \
                    --multiprocessing_distributed true -- \
                    ${python} -m espnet2.bin.lm_train \
                    --ngpu "${ngpu}" \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}" \
                    --token_list "${token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --valid_data_path_and_name_and_type "${lm_trn_txt},text,text" \
                    --valid_shape_file "${lm_stats_dir}/${uttid}/valid/text_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --output_dir "${lm_exp}/${uttid}" \
                    ${_opts} ${lm_args}
            done
        fi

        if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
            log "Stage 9: Calc perplexity"
            _opts=
            for dir in data/lm_train/*; do
                uttid=${dir##*/}
                lm_trn_txt=data/lm_train/${uttid}/"lm_train.txt"
                log "Perplexity calculation started... log: '${lm_exp}/${uttid}/perplexity_test/lm_calc_perplexity.log'"
                # shellcheck disable=SC2086
                ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}/${uttid}"/perplexity_test/lm_calc_perplexity.log \
                    ${python} -m espnet2.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_trn_txt},text,text" \
                    --train_config "${lm_exp}/${uttid}"/config.yaml \
                    --model_file "${lm_exp}/${uttid}/${inference_lm}" \
                    --output_dir "${lm_exp}/${uttid}/perplexity_test" \
                    ${_opts}
                log "PPL: ${lm_trn_txt}: $(cat ${lm_exp}/${uttid}/perplexity_test/ppl)"
            done
        fi
    else
        log "Stage 7-9: Skip traditional LM training stages: use_biased_lm=${use_biased_lm}, skip_lm_train=${skip_lm_train}, use_lm=${use_lm}..."
    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        if "${use_ngram}"; then
            # Note that ngram language models need the installation of kenlm
            log "Stage 10: Ngram Training"

            ngram_key_dir=data/lm_train/keys
            mlevel_scp_dir=data/lm_train/meeting_scps
            mkdir -p ${ngram_key_dir}
            rm -rf ${mlevel_scp_dir}/*
            mkdir -p ${mlevel_scp_dir}
            ${python} local/prep_ngram_keys.py \
                --input_dir data/lm_train/utts \
                --slevel_keys ${ngram_key_dir}/ngram_keys.scp \
                --mscript_dir ${mlevel_scp_dir} \
                --mlevel_keys ${ngram_key_dir}/mlevel_keys.scp

            # 1. Split the segment level key file
            key_file=${ngram_key_dir}/ngram_keys.scp
            _logdir=${ngram_key_dir}/logdir
            mkdir -p ${_logdir}
            split_scps=""
            _nj=$(min "${nj}" "$(wc <${key_file} -l)")

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Perform ngram training in parallel using kenlm
            ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/ngram_train.JOB.log \
                ${python} local/ngram_train.py \
                --keyfile ${_logdir}/keys.JOB.scp \
                --ngram_exp ${ngram_exp}/slevel \
                --ngram_num ${ngram_num}

            # 1. Split the meeting level key file
            key_file=${ngram_key_dir}/mlevel_keys.scp
            _logdir=${ngram_key_dir}/logdir
            mkdir -p ${_logdir}
            split_scps=""
            _nj=$(min "${nj}" "$(wc <${key_file} -l)")

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/mlevel_keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Perform ngram training in parallel using kenlm
            ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/ngram_mlevel_train.JOB.log \
                ${python} local/ngram_train.py \
                --keyfile ${_logdir}/mlevel_keys.JOB.scp \
                --ngram_exp ${ngram_exp}/mlevel \
                --ngram_num ${ngram_num}
        else
            log "Stage 10: Skip ngram stages: use_ngram=${use_ngram}"
        fi
    fi
else
    log "LM training skipped, using general LM for decoding..."
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Primary Decoding: decode_dir=${align_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi

    _data="${data_feats}/decode"

    if "${use_ngram}"; then
        _opts+="--inference_ngram ${inference_ngram} "
        _opts+="--ngram_dir ${ngram_exp}/mlevel "
    fi

    # 2. Generate run.sh
    log "Generate '${align_exp}/${inference_tag}/run.sh'. You can resume the process from stage 11 using this script"
    mkdir -p "${align_exp}/${inference_tag}"
    echo "${run_args} --stage 11 \"\$@\"; exit \$?" >"${align_exp}/${inference_tag}/run.sh"
    chmod +x "${align_exp}/${inference_tag}/run.sh"

    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${align_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _feats_type="$(<${_data}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
        else
            _scp=feats.scp
            _type=kaldi_ark
        fi

        # 1. Split the key file
        key_file=${_data}/${_scp}
        sorted_key_file=${_data}/${_scp}.sorted
        split_scps=""
        if "${use_k2}"; then
            # Now only _nj=1 is verified if using k2
            _nj=1
        else
            _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")
        fi

        # Re-sort the key_file with the mid as the key for efficient inference (since the model reloads as mid changes)
        sort -t "_" -k 2 -o ${sorted_key_file} ${key_file}

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${sorted_key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            ${python} local/primary_inference.py \
            --batch_size ${batch_size} \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type "${sorted_key_file},speech,${_type}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --asr_train_config "${asr_exp}"/config.yaml \
            --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
            --output_dir "${_logdir}"/output.JOB \
            ${_opts} ${inference_args} || {
            cat $(grep -l -i error "${_logdir}"/asr_inference.*.log)
            exit 1
        }

        # 3. Concatenates the output files from each jobs
        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done
    done
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: Merge decoded text for each utterance."

    dset=decode
    _dir="${align_exp}/${inference_tag}/${dset}"
    mkdir -p ${_dir}/merged
    rm -f ${_dir}/merged/{*.txt,*.sorted}
    ${python} local/merge_txt.py \
        --input ${_dir}/text \
        --output_dir ${_dir}/merged \
        --decode

    # Sort the merged text using -V to fix the insufficient 0-padding issue
    for txt in "${_dir}"/merged/*; do
        fname=${txt##/}
        sorted=${fname}.sorted
        awk 'match($0, /seg[0-9]+/) {print substr($0, RSTART+3, RLENGTH-3) " " $0}' "${txt}" |
            sort -k 1 -V | cut -f 3- -d " " | awk -v d=" " '{s=(NR==1?s:s d)$0}END{print s}' >"${sorted}"
        mv "${sorted}" "${txt}"
    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "Stage 13: Compute the Levenshtein distance of the decoded text and the groud-truth text."
    dset=decode
    _dir="${align_exp}/${inference_tag}/decode"
    raw_anchor_dir="${_dir}/anchors/raw"
    token_tag=char
    _opts=

    # Requires pip install pinyin_jyutping_sentence if --phoneme_align
    if "${phoneme_align}"; then
        token_tag=phoneme
        _opts+="--use_phoneme "
        if "${ignore_tone}"; then
            _opts+="--ignore_tone "
        fi
    fi

    if "${heuristic_search}"; then
        _opts+="--heuristic "
        raw_anchor_dir="${_dir}/anchors_heuristic/raw"
    fi

    token_anchor_dir="${raw_anchor_dir}/${token_tag}"
    mkdir -p "${token_anchor_dir}"
    mkdir -p "${raw_anchor_dir}"/char
    rm -f "${raw_anchor_dir}"/*/*.anchor

    # Requires pip install cn2an for numbers2chars conversion
    to_align_dir=${_dir}/to_align
    mkdir -p ${to_align_dir}

    ${python} local/txt_pre_align.py \
        --decoded_dir ${_dir}/merged \
        --text_map data/${dset}/text_map \
        --output_dir ${to_align_dir} \
        ${_opts}

    key_file=${to_align_dir}/search_map
    _logdir=${to_align_dir}/logdir
    mkdir -p "${_logdir}"
    _nj=$(min "${nj}" "$(wc <${key_file} -l)")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/align_text.JOB.log \
        ${python} local/align_text.py \
        --keyfile "${_logdir}"/keys.JOB.scp \
        --token_type ${token_tag} \
        --to_align_dir ${to_align_dir} \
        --eps "\"${eps}\"" \
        --raw_anchor_dir ${raw_anchor_dir} || {
        cat $(grep -l -i error "${_logdir}"/align_text.*.log)
        exit 1
    }
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    log "Stage 14: Find the best primary alignment between the decoded audio and ref scp."
    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs
    raw_anchor_dir="${_dir}/anchors/raw"
    if "${heuristic_search}"; then
        primary_outputs+="_heuristic"
        raw_anchor_dir="${_dir}/anchors_heuristic/raw"
    fi
    mkdir -p ${primary_outputs}
    to_align_dir=${_dir}/to_align
    stats_dir=${primary_outputs}/stats
    rm -f ${stats_dir}/*
    mkdir -p ${stats_dir}

    ln -s "${PWD}"/${to_align_dir}/stats/* "${PWD}"/${stats_dir}

    # Among the alignment results, find the best match of each decoded audio
    ${python} local/primary_mapping.py \
        --aligned_dir ${raw_anchor_dir}/char \
        --output_dir ${primary_outputs}

    ${python} local/primary_stats.py \
        --dump ${primary_outputs}/dump \
        --groud_truth /home/cxiao7/research/speech2text/align_data_v0/processed/metadata/metadata_map.txt \
        --stats_dir ${stats_dir}
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    log "Stage 15: Prepare the data for segmentation."
    char_anchor_dir=${align_exp}/${inference_tag}/decode/anchors/raw/char
    keys_dir=${align_exp}/${inference_tag}/decode/anchors/keys
    dump_dir=${keys_dir}/dump
    mkdir -p ${dump_dir}
    primary_outputs=${align_exp}/${inference_tag}/decode/primary_outputs
    if "${heuristic_search}"; then
        primary_outputs+="_heuristic"
        char_anchor_dir=${align_exp}/${inference_tag}/decode/anchors_heuristic/raw/char
    fi

    # Generate the key file in the format "<anchor_file_path> <text_file_path>"
    rm -f ${dump_dir}/*/text
    ${python} local/generate_seg_key_file.py \
        --text ${align_exp}/${inference_tag}/decode/text \
        --scp_map ${primary_outputs}/scp_map \
        --anchor_dir ${char_anchor_dir} \
        --output ${keys_dir}/anchors.scp

    # Generate the clip_info key file for segmentation
    ${python} local/generate_clip_info.py \
        --vad_dir ${vad_data_dir}/decode \
        --output ${keys_dir}/clip_info \
        --input_format ${seg_file_format}
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    log "Stage 16: Determine each VAD segment's corresponding ground-truth text."
    anchors_dir=${align_exp}/${inference_tag}/decode/anchors
    keys_dir=${align_exp}/${inference_tag}/decode/anchors/keys
    key_file=${keys_dir}/anchors.scp
    clip_info=${keys_dir}/clip_info
    mkdir -p ${anchors_dir}/outputs

    ${python} local/seg_align.py \
        --key_file ${key_file} \
        --output_dir ${anchors_dir}/outputs \
        --clip_info ${clip_info} \
        --eps "${eps}"
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    log "Stage 17: Merge the aligned data portion."
    align_outputs_dir=${align_exp}/${inference_tag}/decode/anchors/outputs
    export_dir=data/aligned_iter_0
    wav_scp="data/decode/wav.scp"
    mkdir -p ${export_dir}
    primary_outputs=${align_exp}/${inference_tag}/decode/primary_outputs
    if "${heuristic_search}"; then
        primary_outputs+="_heuristic"
    fi

    cat ${align_outputs_dir}/*/segments >${export_dir}/segments
    cat ${align_outputs_dir}/*/text >${export_dir}/text

    ${python} local/filter_aligned_utt.py \
        --scp_map ${primary_outputs}/scp_map \
        --wav_scp ${wav_scp} \
        --output ${export_dir}/wav.scp \
        --utt2spk ${export_dir}/utt2spk

    if "${wrap_primary_results}"; then
        _opts=
        if "${wrap_primary_results_norm}"; then
            _opts+="--dump_dir ${align_exp}/${inference_tag}/decode/to_align/dump "
        else
            _opts+="--text_map data/decode/text_map "
        fi
        wrap_dir=data/primary_results
        mkdir -p ${wrap_dir}
        ${python} local/wrap_pr_text.py \
            --scp_map ${primary_outputs}/scp_map \
            --output ${wrap_dir}/text \
            --utt2spk ${wrap_dir}/utt2spk \
            ${_opts}
        cp ${export_dir}/wav.scp ${wrap_dir}/wav.scp
    fi
fi

if [ ${max_iter} -ge 1 ]; then
    if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
        log "Stage 18: Start bootstrapping for alignment improvement."
    fi
else
    log "Bootstrapping skipped, max_iter=${max_iter}..."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
