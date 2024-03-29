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
input_text_dir="/home/cxiao7/research/speech2text/align_data_v4/processed/txt"
input_audio_dir="/home/cxiao7/research/speech2text/align_data_v4/processed/audio"

# VAD related
vad_data_dir=data/vad
vad_nj=80
vad_mthread=2
seg_file_format="kaldi" # Segment file format: "json", "kaldi"

# Bootstrapping related
max_finetune_iter=1
finetune_asr_config=

# Pre-train related
pretrain_asr=false
pretrain_exp=pretrain_exp
skip_lm_train=false
asr_inference_config=
asr_lm_config=

# Alignment related
use_biased_lm=true
phoneme_align=false
ignore_tone=true
eps="***"
align_exp=
heuristic_search=true
wrap_primary_results=true
wrap_primary_results_norm=true
compute_primary_stats=true
flex_graph_dir=
flex_deletion_weight=0
align_config=
flex_window_size=180
flex_overlap_size=30
flex_insertion_weight=0
filter_wer_threshold=0.95

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
nj=64                # The number of parallel jobs.
inference_nj=64      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

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
lang_dir=data/lang_phone          # Directory for storing the language data, lexicons, etc.

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
graph_dir=

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
phntoken_list="${token_listdir}"/phn/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = phn ]; then
    token_list="${phntoken_list}"
    bpemodel=none
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
if [ -z "${graph_dir}" ]; then
    graph_dir="${align_exp}/graphs"
fi
if [ -z "${flex_graph_dir}" ]; then
    flex_graph_dir="${align_exp}/align_graphs"
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
        --kaldi_style true \
        --stage 1 \
        --stop_stage 2
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

        # # TODO: Remove
        # ${python} local/fix_wavscp.py \
        #     --input_file /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.25 \
        #     --output /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.25.fixed
        # ${python} local/fix_wavscp.py \
        #     --input_file /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.34 \
        #     --output /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.34.fixed
        # ${python} local/fix_wavscp.py \
        #     --input_file /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.36 \
        #     --output /home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/dump_v4/raw/decode/logs/segments.36.fixed

        # for dset in ${test_sets}; do
        #     if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
        #         _suf="/org"
        #     else
        #         _suf=""
        #     fi
        #     # TODO: Fix this
        #     # utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
        #     # rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
        #     _opts=
        #     if [ -e data/"${dset}"/segments ]; then
        #         # "segments" is used for splitting wav files which are written in "wav".scp
        #         # into utterances. The file format of segments:
        #         #   <segment_id> <record_id> <start_time> <end_time>
        #         #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
        #         # Where the time is written in seconds.
        #         _opts+="--segments data/${dset}/segments "
        #     fi
        #     # shellcheck disable=SC2086
        #     scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        #         --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
        #         "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

        #     echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
        # done
    fi
fi

if ! "${skip_lm_train}" && "${use_biased_lm}" || "${pretrain_asr}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Generate LM training text for each utterance"
        mkdir -p data/lm_train/utts
        rm -rf data/lm_train/utts
        # Generate lm_train.txt for each utterance on sentence level
        ${python} local/generate_lm_train_text.py \
            --text_map data/decode/sent_text_map \
            --output_dir data/lm_train/utts \
            --sentence
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = char ] || [ "${token_type}" = word ] || [ "${token_type}" = phn ]; then
            log "Stage 5: Generate merged lm_train.txt for further tokenization if needed"

            # The text will be merged with the pretrain data to form the token list
            ${python} local/merge_lm_train.py \
                --utts_dir data/lm_train/utts \
                --output data/lm_train/lm_train.txt \
                --sentence
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
            --nj ${nj} \
            --inference_nj ${inference_nj} \
            --gpu_inference ${gpu_inference} \
            --use_lm false \
            --use_word_lm false \
            --use_ngram true \
            --use_k2 true \
            --lm_config "${asr_lm_config}" \
            --token_type phn \
            --feats_type raw \
            --speed_perturb_factors "${speed_perturb_factors}" \
            --asr_config "${asr_config}" \
            --asr_exp ${asr_exp} \
            --inference_config "${asr_inference_config}" \
            --train_set "${asr_train_set}" \
            --valid_set "${asr_train_dev}" \
            --test_sets "${asr_test_set}" \
            --lm_train_text "data/${asr_train_set}/text" "$@" \
            --expdir ${pretrain_exp} \
            --add_token_text data/lm_train/lm_train.txt \
            --dumpdir ${dumpdir} \
            --g2p "jyutping" \
            --lang_dir ${lang_dir} \
            --stage 15 \
            --stop_stage 15
    fi
fi

if ! "${skip_lm_train}" && "${use_biased_lm}" && [ ${stage} -le 10 ] && [ ${stop_stage} -ge 7 ]; then
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
                --mscript_dir ${mlevel_scp_dir} \
                --mlevel_keys ${ngram_key_dir}/mlevel_keys.scp

            # No longer support segment level n-gram model training as the runtime cost is prohibitive for the full set
            # # 1. Split the segment level key file
            # key_file=${ngram_key_dir}/ngram_keys.scp
            # _logdir=${ngram_key_dir}/logdir
            # mkdir -p ${_logdir}
            # split_scps=""
            # _nj=$(min "${nj}" "$(wc <${key_file} -l)")

            # for n in $(seq "${_nj}"); do
            #     split_scps+=" ${_logdir}/keys.${n}.scp"
            # done
            # # shellcheck disable=SC2086
            # utils/split_scp.pl "${key_file}" ${split_scps}

            # # 2. Perform ngram training in parallel using kenlm
            # ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/ngram_train.JOB.log \
            #     ${python} local/ngram_train.py \
            #     --keyfile ${_logdir}/keys.JOB.scp \
            #     --ngram_exp ${ngram_exp}/slevel \
            #     --ngram_num ${ngram_num}

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

if [ "${token_type}" = phn ]; then
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Preparing decoding graphs in ${graph_dir}"

        # In principle the lexicon FST is compiled in pretraining stage
        if [ ! -f "${lang_dir}"/L_disambig.pt ]; then
            # The <blank> symbol is removed, as it will be replaced by <eps> anyway
            tail -n +2 ${lang_dir}/raw_lexicon.txt >${lang_dir}/lexicon.txt

            ${python} ./local/prepare_lang.py \
                --lang_dir ${lang_dir} \
                --token_list ${token_list} \
                --sil_prob 0 \
                --eps "<blank>"
        fi

        # (Cihan): This could be parallelized but it turns out that it's not very time consuming
        for _dir in "${ngram_exp}"/mlevel/*; do
            mid=${_dir##*/}
            _ngram_exp=${ngram_exp}/mlevel/${mid}
            _graph_dir=${graph_dir}/${mid}
            mkdir -p "${_graph_dir}"

            # use "-" instead of "_" for kaldilm
            ${python} -m kaldilm \
                --read-symbol-table="${lang_dir}/words.txt" \
                --disambig-symbol="#0" \
                --max-order="${ngram_num}" \
                "${_ngram_exp}/${ngram_num}gram.arpa" >"${_graph_dir}/G_${ngram_num}_gram.fst.txt"

            ${python} ./local/compile_hlg.py \
                --lang_dir "${lang_dir}" \
                --graph_dir "${_graph_dir}" \
                --ngram_num "${ngram_num}"
        done
    fi
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: Primary Decoding: decode_dir=${align_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if "${use_k2}"; then
        primary_inference_tool="local/primary_inference_k2.py"

        use_ngram=false
        _opts+="--graph_dir ${graph_dir} "
        _opts+="--word_token_list ${lang_dir}/words.txt "
        _opts+="--is_ctc_decoding False "
        _opts+="--token_type ${token_type} "
    else
        primary_inference_tool="local/primary_inference.py"
    fi

    if [ -n "${inference_config}" ]; then
        if "${use_k2}"; then
            _opts+="--k2_config ${inference_config} "
        else
            _opts+="--config ${inference_config} "
        fi
    fi

    if "${use_ngram}"; then
        _opts+="--inference_ngram ${inference_ngram} "
        _opts+="--ngram_dir ${ngram_exp}/mlevel "
    fi

    # 2. Generate run.sh
    log "Generate '${align_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
    mkdir -p "${align_exp}/${inference_tag}"
    echo "${run_args} --stage 12 \"\$@\"; exit \$?" >"${align_exp}/${inference_tag}/run.sh"
    chmod +x "${align_exp}/${inference_tag}/run.sh"

    _data="${data_feats}/decode"
    # TODO: Change it back to decode
    # _dir="${align_exp}/${inference_tag}/decode_rerun"
    _dir="${align_exp}/${inference_tag}/decode"
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
    sorted_key_file=${key_file}.sorted
    split_scps=""
    _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

    # Re-sort the key_file with the mid as the key for efficient inference (since the model reloads as mid changes)
    sort -t "_" -k 2 -o ${sorted_key_file} ${key_file}

    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${sorted_key_file}" ${split_scps}

    # 2. Submit decoding jobs
    log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
    # It seems that c12 and b02 are broken for k2
    # shellcheck disable=SC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" -l "hostname=!c12*\&!b02*,mem_free=8G,ram_free=8G" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
        ${python} ${primary_inference_tool} \
        --batch_size ${batch_size} \
        --ngpu "${_ngpu}" \
        --data_path_and_name_and_type "${sorted_key_file},speech,${_type}" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --asr_train_config "${asr_exp}"/config.yaml \
        --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
        --output_dir "${_logdir}"/output.JOB \
        ${_opts} ${inference_args} || {
        cat "$(grep -l -i error ${_logdir}/asr_inference.*.log)"
        exit 1
    }

    # 3. Concatenates the output files from each jobs
    for f in token token_int score text; do
        if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
            for i in $(seq "${_nj}"); do
                if [ -f "${_logdir}/output.${i}/1best_recog/${f}" ]; then
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                fi
            done | sort -k1 >"${_dir}/${f}"
        fi
    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "Stage 13: Merge decoded text for each utterance."

    _dir="${align_exp}/${inference_tag}/decode"
    rm -rf ${_dir}/merged
    mkdir -p ${_dir}/merged
    ${python} local/merge_txt.py \
        --input ${_dir}/token \
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

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    log "Stage 14: Merged text align for primary segmentation"

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

    token_anchor_dir="${raw_anchor_dir}/${token_tag}"
    mkdir -p "${token_anchor_dir}"
    mkdir -p "${raw_anchor_dir}"/char
    rm -f "${raw_anchor_dir}"/*/*.anchor
    to_align_dir=${_dir}/to_align
    mkdir -p ${to_align_dir}

    # Generate sorted key files for decoded text
    ${python} local/sort_decoded_keys.py \
        --decoded_dir ${_dir}/merged \
        --output ${to_align_dir}/decode_text_map

    # Note that for now this cannot be splitted as it might break scps in the middle w.r.t. a meeting
    ${python} local/dp_pre_align.py \
        --text_map data/decode/text_map \
        --output_dir ${to_align_dir} \
        --ref \
        ${_opts}

    ${python} local/dp_pre_align.py \
        --text_map ${to_align_dir}/decode_text_map \
        --output_dir ${to_align_dir} \
        ${_opts}

    ${python} local/build_search_map.py \
        --dir ${to_align_dir}/dump \
        --output ${to_align_dir}/search_map \
        ${_opts}
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    log "Stage 15: Compute the Levenshtein distance of the decoded text and the groud-truth text."

    _dir="${align_exp}/${inference_tag}/decode"
    raw_anchor_dir="${_dir}/anchors/raw"
    to_align_dir=${_dir}/to_align
    key_file=${to_align_dir}/search_map
    _logdir=${to_align_dir}/logdir
    mkdir -p "${_logdir}"
    token_tag=char

    # # phoneme_align is broken for some reason, more phs than chars
    # if "${phoneme_align}"; then
    #     token_tag=phoneme
    # fi

    _nj=$(min "${nj}" "$(wc <${key_file} -l)")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # ${python} local/align_text.py \
    #     --keyfile "${_logdir}"/keys.1.scp \
    #     --token_type ${token_tag} \
    #     --to_align_dir ${to_align_dir} \
    #     --eps "\"${eps}\"" \
    #     --raw_anchor_dir ${raw_anchor_dir}

    ${decode_cmd} --gpu "0" JOB=1:"${_nj}" "${_logdir}"/align_text.JOB.log \
        ${python} local/align_text.py \
        --keyfile "${_logdir}"/keys.JOB.scp \
        --token_type ${token_tag} \
        --to_align_dir ${to_align_dir} \
        --eps "\"${eps}\"" \
        --raw_anchor_dir ${raw_anchor_dir} || {
        cat "$(grep -l -i error ${_logdir}/align_text.*.log)"
        exit 1
    }
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    log "Stage 16 (Experimental): Find anchors for each decoded segment."

    _dir="${align_exp}/${inference_tag}/decode"
    char_dir="${_dir}/anchors/raw/char"
    to_align_dir=${_dir}/to_align
    # search_map=${to_align_dir}/search_map

    primary_outputs=${_dir}/primary_outputs
    _logdir=${primary_outputs}/logdir
    mkdir -p "${_logdir}"

    ${python} local/match_hyp.py \
        --input_dir ${char_dir} \
        --output_dir ${primary_outputs} \
        --ratio_threshold 0.2 \
        --score_threshold 0.4

    # Place sentence breaks for flexible alignment
    mkdir -p ${primary_outputs}/dump/output
    ${python} local/break_sentences.py \
        --input_dir ${primary_outputs}/dump/idx \
        --output_dir ${primary_outputs}/dump/output \
        --text_map data/decode/sent_text_map \
        --decode_text_map "${to_align_dir}"/decode_text_map

    # Export the aligned data
    mkdir -p ${primary_outputs}/export
    ${python} local/export_aligned_data.py \
        --input_dir ${primary_outputs}/dump/output/txt \
        --output_dir ${primary_outputs}/export \
        --wavscp data/decode/wav.scp \
        --utt2spk data/decode/utt2spk
fi

# TODO: Wrap this up with an if
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
    log "Stage 18: Perform re-segmentation for flexible alignment."
    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs

    keyfile=${primary_outputs}/export/wav.scp
    flex_align_dir=${align_exp}/flex_align/decode
    out_dir=${flex_align_dir}/re_seg/raw
    mkdir -p out_dir

    ${python} local/resegmentation.py \
        --keyfile ${keyfile} \
        --output_dir ${out_dir} \
        --vad_dir ${vad_data_dir} \
        --vad \
        --window_size ${flex_window_size} \
        --overlap_size ${flex_overlap_size}

    flex_data=${flex_align_dir}/re_seg/data

    local/data_vad_decode.sh \
        --local_data_dir ${out_dir} \
        --dst ${flex_data} \
        --stage 1 \
        --stop_stage 1 --kaldi_style true

    log "Format wav.scp: ${flex_data}/ -> ${data_feats}/flex_data"

    utils/copy_data_dir.sh --validate_opts --non-print ${flex_data} "${data_feats}/flex_data"
    rm -f ${data_feats}/flex_data/{segments,wav.scp,reco2file_and_channel,reco2dur}
    _opts="--segments ${flex_data}/segments "
    # shellcheck disable=SC2086
    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
        "${flex_data}/wav.scp" "${data_feats}/flex_data"

    echo "${feats_type}" >"${data_feats}/flex_data/feats_type"
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
    log "Stage 19: Building graphs for flexible alignment."

    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs

    # init_export_dir=data/aligned_iter_0/raw
    # TODO: Change
    # init_export_dir=/home/cxiao7/research/speech2text/for_k2/data
    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode

    # Split the wav.scp file and the corresponding text file generated by
    # previous stages
    text_key_file=${init_export_dir}/text_map
    _logdir=${flex_align_dir}/logdir
    mkdir -p "${_logdir}"
    _nj=$(min "${inference_nj}" "$(wc <${text_key_file} -l)")
    split_text_scps=""
    for n in $(seq "${_nj}"); do
        split_text_scps+=" ${_logdir}/text.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${text_key_file}" ${split_text_scps}

    ${decode_cmd} --gpu "0" -l "hostname=!c12*\&!b02*,mem_free=32G,ram_free=32G" JOB=1:"${_nj}" "${_logdir}"/build_fa_graphs.JOB.log \
        ${python} local/build_fa_graphs.py \
        --key_file "${_logdir}"/text.JOB.scp \
        --lang_dir "${lang_dir}" \
        --weight 0 \
        --deletion_weight "${flex_deletion_weight}" \
        --determinize \
        --allow_unk \
        --insertion_weight ${flex_insertion_weight} \
        --output_dir ${flex_graph_dir} || {
        cat "$(grep -l -i error ${_logdir}/build_fa_graphs.*.log)"
        exit 1
    }
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
    log "Stage 20: Perform flexible alignment."

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs

    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode
    _data="${data_feats}/flex_data"

    _opts=
    _opts+="--graph_dir ${flex_graph_dir} "
    _opts+="--word_token_list ${lang_dir}/words.txt "
    _opts+="--is_ctc_decoding False "
    _opts+="--token_type ${token_type} "
    _opts+="--k2_config ${align_config} "

    # Split the wav.scp file and the corresponding text file generated by
    # previous stages
    wav_key_file=${_data}/wav.scp
    _logdir=${flex_align_dir}/align_logdir
    mkdir -p "${_logdir}"
    _nj=$(min "${inference_nj}" "$(wc <${wav_key_file} -l)")
    split_wav_scps=""
    for n in $(seq "${_nj}"); do
        split_wav_scps+=" ${_logdir}/wav.${n}.scp"
    done

    # shellcheck disable=SC2086
    utils/split_scp.pl "${wav_key_file}" ${split_wav_scps}

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

    ${flex_align_cmd} --gpu "${_ngpu}" -l "hostname=!c12*\&!b02*,mem_free=40G,ram_free=40G" JOB=1:"${_nj}" "${_logdir}"/flex_align.JOB.log \
        ${python} local/flex_align.py \
        --batch_size ${batch_size} \
        --ngpu "${_ngpu}" \
        --data_path_and_name_and_type "${wav_key_file},speech,${_type}" \
        --key_file "${_logdir}"/wav.JOB.scp \
        --asr_train_config "${asr_exp}"/config.yaml \
        --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
        --output_dir "${_logdir}"/output.JOB \
        ${_opts} ${inference_args} || {
        cat "$(grep -l -i error ${_logdir}/flex_align.*.log)"
        exit 1
    }

    # 3. Concatenates the output files from each jobs
    for f in token token_int score text alignments; do
        if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}"
            done | sort -k1 >"${flex_align_dir}/${f}"
        fi
    done
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
    log "Stage 21: Rerun failed instances."

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs

    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode
    rerun_dir=${flex_align_dir}/rerun
    _data="${data_feats}/flex_data"
    mkdir -p "${rerun_dir}"

    _opts=
    _opts+="--graph_dir ${flex_graph_dir} "
    _opts+="--word_token_list ${lang_dir}/words.txt "
    _opts+="--is_ctc_decoding False "
    _opts+="--token_type ${token_type} "
    _opts+="--k2_config ${align_config} "

    # Split the wav.scp file and the corresponding text file generated by
    # previous stages
    ref_key_file=${_data}/wav.scp
    _logdir=${rerun_dir}/logdir
    mkdir -p "${_logdir}"

    hyp_key_file=${flex_align_dir}/alignments

    # Scan the combined output file and find the utterances that failed
    wav_key_file=${rerun_dir}/wav.scp

    # ${python} local/generate_failed_keys.py \
    #     --ref_key_file ${ref_key_file} \
    #     --hyp_key_file "${hyp_key_file}" \
    #     --output_key_file "${wav_key_file}"

    _nj=$(min "${inference_nj}" "$(wc <${wav_key_file} -l)")
    split_wav_scps=""
    for n in $(seq "${_nj}"); do
        split_wav_scps+=" ${_logdir}/wav.${n}.scp"
    done

    # # shellcheck disable=SC2086
    # utils/split_scp.pl "${wav_key_file}" ${split_wav_scps}

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

    # # TODO: Remove (rerunning the rerun of job 8 which has the weird label -<a_huge_#> error) for only one instance
    # ${flex_align_cmd} --gpu "${_ngpu}" -l "hostname=!c12*\&!b02*,mem_free=40G,ram_free=40G" JOB="8" "${_logdir}"/rerun.JOB.log \
    #     ${python} local/flex_align.py \
    #     --batch_size ${batch_size} \
    #     --ngpu "${_ngpu}" \
    #     --data_path_and_name_and_type "${wav_key_file},speech,${_type}" \
    #     --key_file "${_logdir}"/wav.JOB.scp \
    #     --asr_train_config "${asr_exp}"/config.yaml \
    #     --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
    #     --output_dir "${_logdir}"/output.JOB \
    #     ${_opts} ${inference_args} || {
    #     cat "$(grep -l -i error ${_logdir}/rerun.8.log)"
    #     exit 1
    # }

    # ${flex_align_cmd} --gpu "${_ngpu}" -l "hostname=!c12*\&!b02*,mem_free=40G,ram_free=40G" JOB=1:"${_nj}" "${_logdir}"/rerun.JOB.log \
    #     ${python} local/flex_align.py \
    #     --batch_size ${batch_size} \
    #     --ngpu "${_ngpu}" \
    #     --data_path_and_name_and_type "${wav_key_file},speech,${_type}" \
    #     --key_file "${_logdir}"/wav.JOB.scp \
    #     --asr_train_config "${asr_exp}"/config.yaml \
    #     --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
    #     --output_dir "${_logdir}"/output.JOB \
    #     ${_opts} ${inference_args} || {
    #     cat "$(grep -l -i error ${_logdir}/rerun.*.log)"
    #     exit 1
    # }

    # # 3. Concatenates the output files from each jobs
    # mkdir -p "${rerun_dir}/output"
    # for f in token token_int score text alignments; do
    #     if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
    #         for i in $(seq "${_nj}"); do
    #             cat "${_logdir}/output.${i}/1best_recog/${f}"
    #         done | sort -k1 >"${rerun_dir}/output/${f}"
    #     fi
    # done

    # 4: Merge the rerun results and sort the file
    # Note that here the score file isn't merged since it's not really useful
    merged_dir=${rerun_dir}/merged
    mkdir -p "${merged_dir}"
    # ${python} local/merge_rerun_results.py \
    #     --ref_key_file ${ref_key_file} \
    #     --to_merge_files "token_int,token,text,alignments" \
    #     --base_dir "${flex_align_dir}" \
    #     --to_merge_dir "${rerun_dir}/output" \
    #     --output_dir "${merged_dir}"

    # 5: Move the previous outputs to a backup directory and the merged file to the original directory
    backup_dir=${flex_align_dir}/backup
    mkdir -p "${backup_dir}"
    for f in token token_int score text alignments; do
        [ -f "${flex_align_dir}/${f}" ] && mv "${flex_align_dir}/${f}" "${backup_dir}/${f}"
        [ -f "${merged_dir}/${f}" ] && cp "${merged_dir}/${f}" "${flex_align_dir}/${f}"
    done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
    log "Stage 22: Form stm files."
    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs

    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode
    out_dir=${flex_align_dir}/outputs
    seg_file=${flex_align_dir}/re_seg/data/segments
    keyfile="${out_dir}"/keys/key.scp
    _logdir=${out_dir}/logdir

    mkdir -p "${_logdir}"
    mkdir -p "${out_dir}"/keys

    # ${python} local/generate_stm_keys.py \
    #     --input_file "${flex_align_dir}/alignments" \
    #     --output "${keyfile}"

    _nj=$(min "${nj}" "$(wc <${keyfile} -l)")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/key.${n}.scp"
    done

    # # shellcheck disable=SC2086
    # utils/split_scp.pl "${keyfile}" ${split_scps}

    # # Note that this algorithm requires sorted wav.scp before flexible alignment
    # ${decode_cmd} --gpu 0 -l "hostname=!c12*\&!b02*,mem_free=40G,ram_free=40G" JOB=1:"${_nj}" "${_logdir}"/clean.JOB.log \
    #     ${python} local/clean_salign.py \
    #     --input_dir "${flex_align_dir}" \
    #     --keyfile "${_logdir}"/key.JOB.scp \
    #     --output_dir "${out_dir}" \
    #     --text_map "${init_export_dir}"/text_map \
    #     --segments "${seg_file}" || {
    #     cat "$(grep -l -i error ${_logdir}/clean.*.log)"
    #     exit 1
    # }

    ${python} local/merge_alignments.py \
        --input_dir "${out_dir}"

    # ${python} local/cal_align_wer.py --input_dir "${out_dir}/data" --textmap ${init_export_dir}/text_map
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
    log "Stage 23: Merge results with the t2t pipeline."
    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs
    wavscp=${primary_outputs}/export/wav.scp

    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode
    output_dir=${align_exp}/output
    # TODO: Formalize this instead of hardcoding
    t2t_dir=/home/cxiao7/research/speech2text/scripts/t2t/exp_v4_3/vecalign/export/data
    datefile=/home/cxiao7/research/speech2text/data/metadata/global/dates.json

    # TODO: Formalize this step, this is a byproduct of a debugging script in the t2t pipeline
    mid_text_dir="/home/cxiao7/research/speech2text/random/sent_check"

    ${python} local/index_raw_stm.py \
        --input_dir "${flex_align_dir}" \
        --output_dir "${output_dir}" \
        --mid_text_dir "${mid_text_dir}"

    ${python} local/export_bistm.py \
        --rstm_dir "${flex_align_dir}/outputs/data" \
        --wavscp "${wavscp}" \
        --idx_dir "${output_dir}/idx" \
        --output_dir "${output_dir}/data" \
        --t2t_dir "${t2t_dir}" \
        --datefile "${datefile}"

    sort -o "${output_dir}/data/asr/text.sorted" "${output_dir}/data/asr/text"
    sort -o "${output_dir}/data/asr/utt2spk.sorted" "${output_dir}/data/asr/utt2spk"
    sort -o "${output_dir}/data/asr/wav.scp.sorted" "${output_dir}/data/asr/wav.scp"
    sort -o "${output_dir}/data/asr/segments.sorted" "${output_dir}/data/asr/segments"
    mv "${output_dir}/data/asr/text.sorted" "${output_dir}/data/asr/text"
    mv "${output_dir}/data/asr/utt2spk.sorted" "${output_dir}/data/asr/utt2spk"
    mv "${output_dir}/data/asr/wav.scp.sorted" "${output_dir}/data/asr/wav.scp"
    mv "${output_dir}/data/asr/segments.sorted" "${output_dir}/data/asr/segments"

    utils/utt2spk_to_spk2utt.pl "${output_dir}/data/asr/utt2spk" >"${output_dir}/data/asr/spk2utt"
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
    log "Stage 24: Prepare data for post-filtering."
    export_dir=${align_exp}/output/data/asr

    log "Format wav.scp: ${export_dir}/ -> ${data_feats}/filter_data"

    utils/copy_data_dir.sh --validate_opts --non-print "${export_dir}" "${data_feats}/filter_data"
    rm -f ${data_feats}/filter_data/{segments,wav.scp,reco2file_and_channel,reco2dur}
    _opts=
    if [ -e "${export_dir}/segments" ]; then
        _opts+="--segments ${export_dir}/segments "
    fi
    # shellcheck disable=SC2086
    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
        "${export_dir}/wav.scp" "${data_feats}/filter_data"

    echo "${feats_type}" >"${data_feats}/filter_data/feats_type"
fi

if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ]; then
    log "Stage 25: Run inference on the data for post-filtering."

    filter_dir=${expdir}/filter

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if "${use_k2}"; then
        primary_inference_tool="local/primary_inference_k2.py"

        use_ngram=false
        _opts+="--graph_dir ${graph_dir} "
        _opts+="--word_token_list ${lang_dir}/words.txt "
        _opts+="--is_ctc_decoding False "
        _opts+="--token_type ${token_type} "
    else
        primary_inference_tool="local/primary_inference.py"
    fi

    if [ -n "${inference_config}" ]; then
        if "${use_k2}"; then
            _opts+="--k2_config ${inference_config} "
        else
            _opts+="--config ${inference_config} "
        fi
    fi

    if "${use_ngram}"; then
        _opts+="--inference_ngram ${inference_ngram} "
        _opts+="--ngram_dir ${ngram_exp}/mlevel "
    fi

    # 2. Generate run.sh
    log "Generate '${filter_dir}/${inference_tag}/run.sh'. You can resume the process from stage 25 using this script"
    mkdir -p "${filter_dir}/${inference_tag}"
    echo "${run_args} --stage 25 \"\$@\"; exit \$?" >"${filter_dir}/${inference_tag}/run.sh"
    chmod +x "${filter_dir}/${inference_tag}/run.sh"

    _data="${data_feats}/filter_data"
    _dir="${filter_dir}/${inference_tag}/decode"
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
    sorted_key_file=${key_file}.sorted
    split_scps=""
    _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

    # Re-sort the key_file with the mid as the key for efficient inference (since the model reloads as mid changes)
    sort -t "_" -k 2 -o ${sorted_key_file} ${key_file}

    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${sorted_key_file}" ${split_scps}

    # 2. Submit decoding jobs
    log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
    # It seems that c12 and b02 are broken for k2
    # shellcheck disable=SC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" -l "hostname=!c12*\&!b02*,mem_free=8G,ram_free=8G" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
        ${python} ${primary_inference_tool} \
        --batch_size ${batch_size} \
        --ngpu "${_ngpu}" \
        --data_path_and_name_and_type "${sorted_key_file},speech,${_type}" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --asr_train_config "${asr_exp}"/config.yaml \
        --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
        --output_dir "${_logdir}"/output.JOB \
        ${_opts} ${inference_args} || {
        cat "$(grep -l -i error ${_logdir}/asr_inference.*.log)"
        exit 1
    }

    # 3. Concatenates the output files from each jobs
    for f in token token_int score text; do
        if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
            for i in $(seq "${_nj}"); do
                if [ -f "${_logdir}/output.${i}/1best_recog/${f}" ]; then
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                fi
            done | sort -k1 >"${_dir}/${f}"
        fi
    done
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ]; then
    log "Stage 26: Calculate WER for the decoding results."
    export_dir=${align_exp}/output/data/asr
    filter_dir=${expdir}/filter
    _dir="${filter_dir}/${inference_tag}/decode"
    _logdir="${filter_dir}/${inference_tag}/logdir"
    mkdir -p "${_logdir}"

    hyp_file=${_dir}/token
    ref_file=${export_dir}/text
    res_file=${_logdir}/aligned.txt
    output_file=${_logdir}/wer.txt

    # Convert text into phone sequence using pinyin_jyutping_sentence
    ${python} local/txt2ph.py \
        --input "${hyp_file}" \
        --output "${_logdir}/hyp.phn"
    ${python} local/txt2ph.py \
        --input "${ref_file}" \
        --output "${_logdir}/ref.phn"

    align-text \
        --special-symbol="${eps}" \
        ark:"${_logdir}/ref.phn" \
        ark:"${_logdir}/hyp.phn" \
        ark,t:- | utils/scoring/wer_per_utt_details.pl \
        --special-symbol="${eps}" >"${res_file}"

    # Compute per-utt WER
    ${python} local/compute_wer.py \
        --input "${res_file}" \
        --output "${output_file}"
fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ]; then
    log "Stage 27: Filter out the instances with high WER and other metrics."
    _data_dir=${align_exp}/output/data
    filter_dir=${expdir}/filter
    output_dir=${filter_dir}/output
    _dir="${filter_dir}/${inference_tag}/decode"
    _logdir="${filter_dir}/${inference_tag}/logdir"
    aligned_file=${_logdir}/aligned.txt
    wer_file=${_logdir}/wer.txt

    # Filter out the instances with high WER
    ${python} local/wer_filter.py \
        --input_dir "${_data_dir}" \
        --output "${output_dir}" \
        --aligned_file "${aligned_file}" \
        --wer "${wer_file}" \
        --threshold "${filter_wer_threshold}" \
        --dumpdir "${_logdir}/dump"

    # TODO: Remove or make it configurable
    /home/cxiao7/research/speech2text/scripts/t2t/utils/calc_seg_len.py
fi

if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ]; then
    log "Stage 28: Create train/dev/test splits."
    filter_dir=${expdir}/filter
    _data_dir=${filter_dir}/output
    split_dir=${expdir}/splits
    mkdir -p "${split_dir}"

    # Create feature files for NACHOS
    _logdir="${filter_dir}/${inference_tag}/logdir"
    out_logdir="${split_dir}/logdir"
    mkdir -p "${out_logdir}"
    ${python} local/get_metadata.py \
        --input_dir "${_logdir}/dump" \
        --data_dir "${_data_dir}" \
        --output_dir "${out_logdir}"
fi

if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ]; then
    log "Stage 29: Gather some statistics of the splits and the whole dataset."
    filter_dir=${expdir}/filter
    _data_dir=${filter_dir}/output
    split_dir=${expdir}/splits
    stat_dir=${split_dir}/stats
    mkdir -p "${stat_dir}"
    spkfile=/home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/align_exp_v4/filter/decode_k2_ngram_ngram_3gram_asr_model_valid.acc.best_use_k2_k2_ctc_decoding_true_use_nbest_rescoring_true/logdir/dump/spk.annot.dump

    ${python} local/get_split_stats.py \
        --data_dir ${split_dir} \
        --annotfile ${spkfile} \
        --output_dir ${stat_dir}
fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ]; then
    log "Stage 30: Replace the Cantonese text with the version with punctuation."
    _dir="${align_exp}/${inference_tag}/decode"
    primary_outputs=${_dir}/primary_outputs
    wavscp=${primary_outputs}/export/wav.scp

    init_export_dir=${primary_outputs}/export
    flex_align_dir=${align_exp}/flex_align/decode
    output_dir=${align_exp}/output_punc
    # TODO: Formalize this instead of hardcoding
    t2t_dir=/home/cxiao7/research/speech2text/scripts/t2t/exp_v4_3/vecalign/export/data
    datefile=/home/cxiao7/research/speech2text/data/metadata/global/dates.json

    # TODO: Formalize this step, this is a byproduct of a debugging script in the t2t pipeline
    punc_text_dir="/home/cxiao7/research/speech2text/random/sent_punc"

    ${python} local/export_bistm.py \
        --rstm_dir "${flex_align_dir}/outputs/data" \
        --punc_text_dir "${punc_text_dir}" \
        --wavscp "${wavscp}" \
        --idx_dir "${output_dir}/idx" \
        --output_dir "${output_dir}/data" \
        --t2t_dir "${t2t_dir}" \
        --datefile "${datefile}"

    sort -o "${output_dir}/data/asr/text.sorted" "${output_dir}/data/asr/text"
    sort -o "${output_dir}/data/asr/utt2spk.sorted" "${output_dir}/data/asr/utt2spk"
    sort -o "${output_dir}/data/asr/wav.scp.sorted" "${output_dir}/data/asr/wav.scp"
    sort -o "${output_dir}/data/asr/segments.sorted" "${output_dir}/data/asr/segments"
    mv "${output_dir}/data/asr/text.sorted" "${output_dir}/data/asr/text"
    mv "${output_dir}/data/asr/utt2spk.sorted" "${output_dir}/data/asr/utt2spk"
    mv "${output_dir}/data/asr/wav.scp.sorted" "${output_dir}/data/asr/wav.scp"
    mv "${output_dir}/data/asr/segments.sorted" "${output_dir}/data/asr/segments"

    utils/utt2spk_to_spk2utt.pl "${output_dir}/data/asr/utt2spk" >"${output_dir}/data/asr/spk2utt"
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ]; then
    log "Stage 31: Filter out the instances with high WER and other metrics."
    _data_dir=${align_exp}/output_punc/data
    filter_dir=${expdir}/filter
    output_dir=${filter_dir}/output_punc
    _dir="${filter_dir}/${inference_tag}/decode"
    _logdir="${filter_dir}/${inference_tag}/logdir"
    aligned_file=${_logdir}/aligned.txt
    wer_file=${_logdir}/wer.txt

    # Filter out the instances with high WER
    ${python} local/wer_filter.py \
        --input_dir "${_data_dir}" \
        --output "${output_dir}" \
        --aligned_file "${aligned_file}" \
        --wer "${wer_file}" \
        --threshold "${filter_wer_threshold}"

    /home/cxiao7/research/speech2text/scripts/t2t/utils/calc_seg_len.py \
        --input "${output_dir}/asr/segments"

fi

if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ]; then
    log "Stage 32: Gather some statistics of the splits and the whole dataset."
    split_dir=${expdir}/splits
    stat_dir=${split_dir}/stats
    mkdir -p "${stat_dir}"
    spkfile=/home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/align_exp_v4/filter/decode_k2_ngram_ngram_3gram_asr_model_valid.acc.best_use_k2_k2_ctc_decoding_true_use_nbest_rescoring_true/logdir/dump/spk.annot.dump

    ${python} local/get_split_stats.py \
        --data_dir ${split_dir} \
        --annotfile ${spkfile} \
        --output_dir ${stat_dir} \
        --punc
fi

if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ]; then
    log "Stage 33: Perform subsplit of the dev datasets to reduce their size."
    split_dir=${expdir}/splits
    tgt_split_dir=${expdir}/subsplits
    mkdir -p "${tgt_split_dir}"
    spkfile=/home/cxiao7/research/espnet/egs2/commonvoice_align/asr1/align_exp_v4/filter/decode_k2_ngram_ngram_3gram_asr_model_valid.acc.best_use_k2_k2_ctc_decoding_true_use_nbest_rescoring_true/logdir/dump/spk.annot.dump

    ${python} local/subsplit.py \
        --data_dir ${split_dir} \
        --annotfile ${spkfile} \
        --output_dir ${split_dir} \
        --punc
fi

# if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
#     log "Stage 19: Start bootstrapping for alignment improvement."
#     primary_outputs=${align_exp}/${inference_tag}/decode/primary_outputs
#     if "${heuristic_search}"; then
#         primary_outputs+="_heuristic"
#     fi

#     ./finetune.sh \
#         --start_iter 0 \
#         --max_iter ${max_finetune_iter} \
#         --num_nodes ${num_nodes} \
#         --expdir ${expdir} \
#         --ngpu ${ngpu} \
#         --nj ${nj} \
#         --gpu_inference ${gpu_inference} \
#         --inference_config "${inference_config}" \
#         --inference_ngram ${inference_ngram} \
#         --inference_asr_model ${inference_asr_model} \
#         --inference_nj ${inference_nj} \
#         --batch_size ${batch_size} \
#         --decode_dumpdir ${data_feats}/decode \
#         --asr_text_fold_length ${asr_text_fold_length} \
#         --speed_perturb_factors "${speed_perturb_factors}" \
#         --dumpdir ${dumpdir} \
#         --token_type ${token_type} \
#         --feats_type ${feats_type} \
#         --asr_config "${finetune_asr_config}" \
#         --token_list ${token_list} \
#         --pretrained_model "${asr_exp}/${inference_asr_model}" \
#         --ngram_exp "${ngram_exp}" \
#         --scp_map ${primary_outputs}/scp_map \
#         --eps "${eps}" \
#         --phoneme_align ${phoneme_align} \
#         --ignore_tone ${ignore_tone} \
#         --heuristic_search ${heuristic_search} \
#         --text_map data/decode/text_map \
#         --vad_data_dir ${vad_data_dir} \
#         --seg_file_format ${seg_file_format} \
#         --python ${python} \
#         --unsegged_decode_wav_scp "data/decode/wav.scp" \
#         --segged_decode_wav_scp "${data_feats}/decode/wav.scp" \
#         --wrap_primary_results ${wrap_primary_results} \
#         --wrap_primary_results_norm ${wrap_primary_results_norm} \
#         --norm_text_dump "${align_exp}/${inference_tag}/decode/to_align/dump" \
#         --stage 12 \
#         --stop_stage 12
# fi

log "Successfully finished. [elapsed=${SECONDS}s]"
