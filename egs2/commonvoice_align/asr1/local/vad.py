import argparse
import json
import torch
from pathlib import Path
import os
from clip_segments import clip_single_audio
from pydub import AudioSegment
from utils import mkdir_if_not_exist
import re
import logging

punc = re.compile(r'\(|\)|\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?')


def labels2sec(labels, fs, uttid):
    res = {}
    for i, label in enumerate(labels):
        res[f"{uttid}_seg{i+1:03}"] = {"start": label["start"]
                                       / fs, "end": label["end"] / fs}

    return res


def single_audio_vad(audio, model, fs, output_dir, read_audio, get_speech_timestamps, concatenate=False, clip=False, kaldi_output=False):
    logging.info(f"Processing {audio}...")
    uttid = re.subn(punc, '', Path(audio).stem.replace(" ", ""))[0]
    seg_dir = os.path.join(output_dir, uttid)
    mkdir_if_not_exist(seg_dir)
    seg_fname = os.path.join(seg_dir, uttid + ".json")

    wav = read_audio(audio, sampling_rate=fs)
    speech_labels = get_speech_timestamps(wav, model, sampling_rate=fs)
    segments = labels2sec(speech_labels, fs, uttid)

    # if args.concatenate:
    #     save_audio('only_speech.wav',
    #                collect_chunks(speech_labels, wav), sampling_rate=fs)

    with open(seg_fname, "w") as f:
        json.dump(segments, f)

    if kaldi_output:
        with open(os.path.join(seg_dir, "segments"), "w") as f:
            for segid, timestamp in segments.items():
                print(
                    f"{segid} {uttid} {timestamp['start']:.2f} {timestamp['end']:.2f}", file=f)

        with open(os.path.join(seg_dir, "wav.scp"), "w") as f:
            print(f"{uttid} {audio}", file=f)

    if clip:
        clip_dir = os.path.join(seg_dir, "clips")
        mkdir_if_not_exist(clip_dir)
        audio_raw = AudioSegment.from_file(audio)
        for segid, timestamp in segments.items():
            seg = timestamp
            seg["segid"] = segid
            seg["audio"] = audio_raw
            seg["output_dir"] = clip_dir
            clip_single_audio(**seg)

    logging.info(f"Successfully processed {audio}.")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze input wave-file and save detected speech interval to json file.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir',
                       type=Path,
                       help='The directory containing the input audio files.')
    group.add_argument(
        "--keyfile",
        type=Path,
        help="The key file (to the audio) for vad sementation."
    )
    parser.add_argument('--output_dir',
                        type=Path,
                        required=True,
                        help='The directory in which the output files (json, concatenation wav, clipped wavs) are stored.')
    parser.add_argument("--fs",
                        type=int,
                        default=16000,
                        help="Sampling Frequency.")
    parser.add_argument('--concatenate',
                        action='store_true',
                        help='If option provided, the segments will be concatenated and stored.')
    parser.add_argument('--clip',
                        action='store_true',
                        help='If option provided, the audio will be clipped and stored.')
    parser.add_argument("--mthread",
                        type=int,
                        default=1,
                        help="Number of threads used for VAD.")
    parser.add_argument('--kaldi_output', action='store_true',
                        help='If true, the script will output a kaldi-style "segments" file')
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    fs = args.fs

    torch.set_num_threads(args.mthread)

    USE_ONNX = False

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=USE_ONNX)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    if args.input_dir:
        for audio in os.listdir(args.input_dir):
            full_path = os.path.join(args.input_dir, audio)
            single_audio_vad(full_path, model, fs, args.output_dir, read_audio, get_speech_timestamps,
                             args.concatenate, args.clip, args.kaldi_output)
    elif args.keyfile:
        with open(args.keyfile, "r") as f:
            for line in f:
                uttid, utt_fp = line.strip().split(" ", maxsplit=1)
                single_audio_vad(utt_fp, model, fs, args.output_dir, read_audio, get_speech_timestamps,
                                 args.concatenate, args.clip, args.kaldi_output)
