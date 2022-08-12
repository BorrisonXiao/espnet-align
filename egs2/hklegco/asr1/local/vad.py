import argparse
import json
from pydoc import cli
import torch
from pathlib import Path
import os
from clip_segments import clip_single_audio
from pydub import AudioSegment


def mkdir_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def labels2sec(labels, fs):
    res = []
    for i, label in enumerate(labels):
        res.append(
            {"segid": f"seg{i+1:03}", "start": label["start"] / fs, "end": label["end"] / fs})

    return res


def single_audio_vad(audio, model, fs, output_dir, read_audio, get_speech_timestamps, concatenate=False, clip=False):
    uttid = Path(audio).stem
    seg_dir = os.path.join(output_dir, uttid)
    mkdir_if_not_exist(seg_dir)
    seg_fname = os.path.join(seg_dir, uttid + ".json")

    wav = read_audio(audio, sampling_rate=fs)
    speech_labels = get_speech_timestamps(wav, model, sampling_rate=fs)
    segments = labels2sec(speech_labels, fs)

    # if args.concatenate:
    #     save_audio('only_speech.wav',
    #                collect_chunks(speech_labels, wav), sampling_rate=fs)

    with open(seg_fname, "w") as f:
        json.dump(segments, f)

    if clip:
        clip_dir = os.path.join(seg_dir, "clips")
        mkdir_if_not_exist(clip_dir)
        audio_raw = AudioSegment.from_file(audio)
        for segment in segments:
            segment["uttid"] = uttid
            segment["audio"] = audio_raw
            segment["output_dir"] = clip_dir
            clip_single_audio(**segment)

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze input wave-file and save detected speech interval to json file.')
    parser.add_argument('--input_dir',
                        type=Path,
                        required=True,
                        help='The directory containing the input audio files.')
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
    args = parser.parse_args()

    fs = args.fs

    torch.set_num_threads(1)

    USE_ONNX = False

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=USE_ONNX)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    for audio in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, audio)
        single_audio_vad(full_path, model, fs, args.output_dir, read_audio, get_speech_timestamps,
                         args.concatenate, args.clip)
