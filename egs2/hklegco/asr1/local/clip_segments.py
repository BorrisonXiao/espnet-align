import argparse
import os
import logging
from pathlib import Path
from pydub import AudioSegment


def read_segment_file(input_filename):
    res = []
    with open(input_filename, "r") as f:
        for line in f:
            parsed_line = line.strip().split(" ")
            segid, uttid = parsed_line[:2]
            start_time, end_time = float(parsed_line[2]), float(parsed_line[3])
            res.append({"segid": segid, "uttid": uttid,
                       "start": start_time, "end": end_time})
    return res


def clip_single_audio(segid, uttid, start, end, audio, output_dir):
    out_fname = "_".join([uttid, segid + ".wav"])
    audio_clip = audio[start * 1000: end * 1000]
    audio_clip.export(os.path.join(output_dir, out_fname), format="wav")


def clip_audios(seg_file_dir, audio_dir, output_dir):
    for seg_file in os.listdir(seg_file_dir):
        seg_info = read_segment_file(os.path.join(seg_file_dir, seg_file))
        clip_fname_base = seg_info[0]["uttid"]
        audio_fname_base = clip_fname_base.split("_", maxsplit=1)[1]
        audio_file = os.path.join(audio_dir, audio_fname_base + ".mp3")
        audio = AudioSegment.from_file(audio_file)

        for info in seg_info:
            info["audio"] = audio
            info["output_dir"] = output_dir

            # TODO: Parallelize this
            clip_single_audio(**info)


def main():
    parser = argparse.ArgumentParser(
        description='Clip the audio file based on the segment file')
    parser.add_argument('--seg_file_dir', type=Path, required=True,
                        help='Input: The directory with segment files.')
    parser.add_argument('--audio_dir', type=Path, required=True,
                        help='Output: The directory in which the original audio files are stored.')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Output: The directory to store the clipped audio files.')
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    clip_audios(args.seg_file_dir, args.audio_dir, args.output_dir)


if __name__ == "__main__":
    main()
