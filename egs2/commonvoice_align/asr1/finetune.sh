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
start_iter=0
max_iter=5
ngpu=4
nj=64
gpu_inference=false
speed_perturb_factors="0.9 1.0 1.1"
dumpdir=dump
token_type=word
inference_nj=64
feats_type=raw
python=python3
expdir=align_exp
audio_format=wav     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0   # Minimum duration in second.
max_wav_duration=200 # Maximum duration in second.
num_nodes=1          # The number of nodes.
stage=10
stop_stage=10

# ASR Model related
asr_config=
asr_tag=
asr_args=
token_list=
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
pretrained_model=          # Pretrained model to load
ignore_init_mismatch=false # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.
asr_text_fold_length=150   # fold_length for text data during ASR training.

# Language Model related
ngram_exp= # The pretrained ngram language models' directory for both meeting level (mlevel) and segment level (slevel)

# Inference related
inference_config=
inference_ngram=
inference_tag=
inference_asr_model=
inference_args=
batch_size=1

# Alignment related
scp_map=
decode_dumpdir=
eps="***"
heuristic_search=false
phoneme_align=true
ignore_tone=true
text_map=
vad_data_dir=
seg_file_format=
unsegged_decode_wav_scp=
segged_decode_wav_scp=
wrap_primary_results=
wrap_primary_results_norm=
norm_text_dump=

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats_base=${dumpdir}/raw
else
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    asr_tag+="_${token_type}"
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
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

    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

for ((iter = start_iter; iter < max_iter; iter++)); do
    data_dir=data/aligned_iter_${iter}
    train_set="train"
    dev_set="dev"
    test_set="test"
    data_feats=${data_feats_base}/finetune/iter_${iter}
    mkdir -p ${data_feats}
    _expdir=${expdir}/finetune/iter_${iter}
    asr_stats_dir="${_expdir}/asr_stats_${feats_type}_${token_type}"
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
    asr_exp="${_expdir}/asr_${asr_tag}"
    align_exp=${_expdir}/align_exp
    _pretrained_model=${pretrained_model}
    if [ "${iter}" -ne 0 ]; then
        _prev_iter=$((iter + 4))
        _pretrained_model="${expdir}/finetune/iter_${_prev_iter}/${inference_asr_model}"
    fi

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        log "Iteration ${iter} stage 0: Perform train/dev/test split on the data"
        mkdir -p ${data_dir}/${train_set}
        mkdir -p ${data_dir}/${dev_set}
        mkdir -p ${data_dir}/${test_set}

        ${python} local/trn_dev_test_split.py \
            --train_dir ${data_dir}/${train_set} \
            --dev_dir ${data_dir}/${dev_set} \
            --test_dir ${data_dir}/${test_set} \
            --data_dir ${data_dir}/raw \
            --ratio "0.7 0.2 0.1"

        for dset in "${train_set}" "${dev_set}" ${test_set}; do
            sort -o ${data_dir}/${dset}/wav.scp ${data_dir}/${dset}/wav.scp.unsorted
            rm ${data_dir}/${dset}/wav.scp.unsorted
            sort -o ${data_dir}/${dset}/text ${data_dir}/${dset}/text.unsorted
            rm ${data_dir}/${dset}/text.unsorted
            sort -o ${data_dir}/${dset}/utt2spk ${data_dir}/${dset}/utt2spk.unsorted
            rm ${data_dir}/${dset}/utt2spk.unsorted
            utils/utt2spk_to_spk2utt.pl ${data_dir}/${dset}/utt2spk >${data_dir}/${dset}/spk2utt
            utils/validate_data_dir.sh --no-feats ${data_dir}/${dset} || exit 1

            ${python} local/calc_audio_length.py \
                --wav_scp ${data_dir}/${dset}/wav.scp \
                --tag ${dset}
        done
    fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
            log "Iteration ${iter} stage 1: Speed perturbation: ${data_dir}/${train_set} -> data/${train_set}_sp"
            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    scripts/utils/perturb_data_dir_speed.sh "${factor}" "${data_dir}/${train_set}" "${data_dir}/${train_set}_sp${factor}"
                    _dirs+="${data_dir}/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="${data_dir}/${train_set} "
                fi
            done
            utils/combine_data.sh "${data_dir}/${train_set}_sp" ${_dirs}
        else
            log "Skip iteration ${iter} stage 1: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Iteration ${iter} stage 2: Format wav.scp: ${data_dir}/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${dev_set}" ${test_set}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${dev_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print ${data_dir}/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e ${data_dir}/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments ${data_dir}/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "${data_dir}/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            done
        fi
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Iteration ${iter} stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${dev_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                awk <"${data_feats}/org/${dset}/utt2num_samples" \
                    -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
                utils/filter_scp.pl <"${data_feats}/org/${dset}/wav.scp" \
                    "${data_feats}/${dset}/utt2num_samples" \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(grep <conf/fbank.conf -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(grep <conf/fbank.conf frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                awk <"${data_feats}/org/${dset}/feats_shape" -F, ' { print $1 } ' |
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                utils/filter_scp.pl <"${data_feats}/org/${dset}/feats.scp" \
                    "${data_feats}/${dset}/feats_shape" \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            awk <"${data_feats}/org/${dset}/text" \
                ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            # utils/fix_data_dir.sh "${data_feats}/${dset}"
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${dev_set}"
        log "Iteration ${iter} stage 4: ASR collect stats: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _opts+="--init_param ${_pretrained_model} "

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        fi

        # 1. Split the key file
        _logdir="${asr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(wc <${_asr_train_dir}/${_scp} -l)" "$(wc <${_asr_valid_dir}/${_scp} -l)")

        key_file="${_asr_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_asr_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${asr_stats_dir}"
        echo "${run_args} --stage 4 \"\$@\"; exit \$?" >"${asr_stats_dir}/run.sh"
        chmod +x "${asr_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.asr_train \
            --collect_stats true \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} || {
            cat $(grep -l -i error "${_logdir}"/stats.*.log)
            exit 1
        }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        awk <"${asr_stats_dir}/train/text_shape" \
            -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/train/text_shape.${token_type}"

        awk <"${asr_stats_dir}/valid/text_shape" \
            -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/valid/text_shape.${token_type}"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${dev_set}"
        log "Iteration ${iter} stage 5: ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((asr_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${asr_speech_fold_length}"
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                    --scps \
                    "${_asr_train_dir}/${_scp}" \
                    "${_asr_train_dir}/text" \
                    "${asr_stats_dir}/train/speech_shape" \
                    "${asr_stats_dir}/train/text_shape.${token_type}" \
                    --num_splits "${num_splits_asr}" \
                    --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
        fi

        log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${asr_exp}"
        echo "${run_args} --stage 5 \"\$@\"; exit \$?" >"${asr_exp}/run.sh"
        chmod +x "${asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ASR training started... log: '${asr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &>/dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${asr_exp})"
        else
            jobname="${asr_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${asr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${asr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.asr_train \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
            --resume false \
            --init_param ${_pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --fold_length "${_fold_length}" \
            --fold_length "${asr_text_fold_length}" \
            --output_dir "${asr_exp}" \
            ${_opts} ${asr_args}
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Iteration ${iter} stage 6: Decoding: decode_dir=${align_exp}"

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

        # 2. Generate run.sh
        log "Generate '${align_exp}/${inference_tag}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${align_exp}/${inference_tag}"
        echo "${run_args} --stage 6 \"\$@\"; exit \$?" >"${align_exp}/${inference_tag}/run.sh"
        chmod +x "${align_exp}/${inference_tag}/run.sh"

        _data=${decode_dumpdir}
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
        # The sorted key_file should already be generated in pre-finetuning pipeline
        sorted_key_file=${key_file}.sorted
        split_scps=""
        _nj=$(min "${inference_nj}" "$(wc <${sorted_key_file} -l)")

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${sorted_key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/finetune_inference.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/finetune_inference.JOB.log \
            ${python} local/finetune_inference.py \
            --batch_size ${batch_size} \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type "${sorted_key_file},speech,${_type}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --asr_train_config "${asr_exp}"/config.yaml \
            --asr_model_file "${asr_exp}/${inference_asr_model}" \
            --scp_map ${scp_map} \
            --inference_ngram ${inference_ngram} \
            --ngram_dir ${ngram_exp} \
            --output_dir "${_logdir}"/output.JOB \
            ${_opts} ${inference_args} || {
            cat $(grep -l -i error "${_logdir}"/finetune_inference.*.log)
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
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Iteration ${iter} stage 7: Merge decoded text for each utterance."

        _dir="${align_exp}/${inference_tag}/decode"
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

    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Iteration ${iter} stage 8: Compute the Levenshtein distance of the decoded text and the groud-truth text."
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
        mkdir -p "${to_align_dir}"

        ${python} local/txt_pre_align.py \
            --decoded_dir "${_dir}/merged" \
            --text_map "${text_map}" \
            --output_dir "${to_align_dir}" \
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
            --to_align_dir "${to_align_dir}" \
            --eps "\"${eps}\"" \
            --raw_anchor_dir "${raw_anchor_dir}" || {
            cat $(grep -l -i error "${_logdir}"/align_text.*.log)
            exit 1
        }
    fi

    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        # TODO: Previously matched documents should not be re-matched?
        log "Iteration ${iter} stage 9: Find the best primary alignment between the decoded audio and ref scp."
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

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Iteration ${iter} stage 10: Prepare the data for segmentation."
        char_anchor_dir=${align_exp}/${inference_tag}/decode/anchors/raw/char
        keys_dir=${align_exp}/${inference_tag}/decode/anchors/keys
        dump_dir=${keys_dir}/dump
        mkdir -p "${dump_dir}"
        primary_outputs=${align_exp}/${inference_tag}/decode/primary_outputs
        if "${heuristic_search}"; then
            primary_outputs+="_heuristic"
            char_anchor_dir=${align_exp}/${inference_tag}/decode/anchors_heuristic/raw/char
        fi

        # Generate the key file in the format "<anchor_file_path> <text_file_path>"
        rm -f "${dump_dir}/*/text"
        ${python} local/generate_seg_key_file.py \
            --text "${align_exp}/${inference_tag}/decode/text" \
            --scp_map ${primary_outputs}/scp_map \
            --anchor_dir "${char_anchor_dir}" \
            --output "${keys_dir}/anchors.scp"

        # Generate the clip_info key file for segmentation
        ${python} local/generate_clip_info.py \
            --vad_dir "${vad_data_dir}/decode" \
            --output "${keys_dir}/clip_info" \
            --input_format "${seg_file_format}"
    fi

    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Iteration ${iter} stage 11: Determine each VAD segment's corresponding ground-truth text."
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

    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Wrap the aligned data portion."
        align_outputs_dir=${align_exp}/${inference_tag}/decode/anchors/outputs
        _next_iter=$((iter + 1))
        _export_dir=data/aligned_iter_${_next_iter}/raw
        mkdir -p ${_export_dir}
        primary_outputs=${align_exp}/${inference_tag}/decode/primary_outputs
        if "${heuristic_search}"; then
            primary_outputs+="_heuristic"
        fi

        cat "${align_outputs_dir}"/*/text >${_export_dir}/text.raw

        # Generates the wav.scp and utt2spk files while renaming the segids
        # with the spkid as prefix all related file to comply with kaldi
        # TODO: Only append spkid prefix if it's not already there
        ${python} local/filter_aligned_utt.py \
            --scp_map ${primary_outputs}/scp_map \
            --wav_scp "${segged_decode_wav_scp}" \
            --output_dir ${_export_dir}
        rm -f ${_export_dir}/text.raw

        sort -o ${_export_dir}/wav.scp ${_export_dir}/wav.scp.unsorted
        rm ${_export_dir}/wav.scp.unsorted
        sort -o ${_export_dir}/utt2spk ${_export_dir}/utt2spk.unsorted
        rm ${_export_dir}/utt2spk.unsorted
        sort -o ${_export_dir}/text ${_export_dir}/text.unsorted
        rm ${_export_dir}/text.unsorted
        utils/utt2spk_to_spk2utt.pl ${_export_dir}/utt2spk >${_export_dir}/spk2utt
        utils/validate_data_dir.sh --no-feats ${_export_dir} || exit 1

        if "${wrap_primary_results}"; then
            _opts=
            if "${wrap_primary_results_norm}"; then
                _opts+="--dump_dir ${norm_text_dump} "
            else
                _opts+="--text_map ${text_map} "
            fi
            wrap_dir=data/primary_results_iter_${iter}
            mkdir -p ${wrap_dir}
            rm -f ${wrap_dir}/*
            ${python} local/wrap_pr_text.py \
                --scp_map ${primary_outputs}/scp_map \
                --wav_scp "${unsegged_decode_wav_scp}" \
                --output_dir ${wrap_dir} \
                ${_opts}

            sort -o ${wrap_dir}/wav.scp ${wrap_dir}/wav.scp.unsorted
            rm ${wrap_dir}/wav.scp.unsorted
            sort -o ${wrap_dir}/text ${wrap_dir}/text.unsorted
            rm ${wrap_dir}/text.unsorted
            sort -o ${wrap_dir}/utt2spk ${wrap_dir}/utt2spk.unsorted
            rm ${wrap_dir}/utt2spk.unsorted
            utils/utt2spk_to_spk2utt.pl ${wrap_dir}/utt2spk >${wrap_dir}/spk2utt
            utils/validate_data_dir.sh --no-feats ${wrap_dir} || exit 1
        fi
    fi
done
