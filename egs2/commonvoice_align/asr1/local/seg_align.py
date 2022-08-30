import argparse
import os
from pathlib import Path
import json
from utils import mkdir_if_not_exist, read_anchor_file, parse_segments_file


def read_seg_text(file):
    """
    Read the text files.
    @return: A list of (segid, text) tuples.
    """
    res = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip().split(" ", maxsplit=1)
            if len(line) <= 1:
                continue
            segid, text = line
            res.append((segid, text))
    return res


def satisfy_match_conditions(ref, hyp, op):
    """
    Determine if two strings can be regarded as a match.
    The conditions can be tuned empirically.
    """
    matched_chars = op.count("C")
    # The percentage/absolute number of matched chars in two strings
    if matched_chars / len(op) >= 0.6:
        return True
    # The number of absolute matched chars in the two strings
    if matched_chars >= 8:
        return True
    criteria = ["C"] * 4
    return any([op[i: i + len(criteria)] == criteria for i in range(len(op) - len(criteria))])


def write_anchor_dump(anchor_segs, aligned_ratio, output):
    """
    Write the anchors into a dump file fur further analysis.
    """
    with open(output, "w") as f:
        print(f"{aligned_ratio:.2}", file=f)
        for seg in anchor_segs:
            print(seg["segid"], file=f)
            print("hyp\t" + "\t".join(seg["hyp"]), file=f)
            print("ref\t" + "\t".join(seg["ref"]), file=f)
            print("ops\t" + "\t".join(seg["ops"]), file=f)


def write_anchor_seg(anchor_segs, seg_info, output, format):
    """
    Write the determined anchors into a segments file.
    """
    res = {}
    if format == "json":
        for anchor in anchor_segs:
            res[anchor["segid"]] = seg_info[anchor["segid"]]
        with open(output, "w") as f:
            json.dump(res, f)
    elif format == "kaldi":
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupport segmentation file format {format}")


def write_text(anchor_segs, output, eps):
    """
    Write the determined anchors into a segments file.
    """
    with open(output, "w") as f:
        for anchor in anchor_segs:
            ref = [char for char in anchor['ref'] if char != eps]
            scp = " ".join(ref)
            print(f"{anchor['segid']} {scp}", file=f)


def seg_align(key_file, output_dir, clip_info, eps):
    """
    Perform the audio-text alignment based on the (segmented-decoded) text2text alignment result.
    """
    # Read the "<uttid> <clip_info_filepath>" key file (either kaldi style or json)
    uttid2clip = {}
    with open(clip_info, "r") as f:
        for line in f:
            clip_uttid, clip_fp = line.strip().split(" ")
            uttid2clip[clip_uttid] = clip_fp
    seg_format = "json" if uttid2clip[list(uttid2clip.keys())[0]].endswith(
        ".json") else "kaldi"

    with open(key_file, "r") as f:
        for line in f:
            anchor_segs = []
            anchor_file, text_file = line.strip().split(" ")
            ref, hyp, op, csid = read_anchor_file(anchor_file)
            seg_text = read_seg_text(text_file)

            seg_idx = 0
            curr_seg_text = seg_text[seg_idx][1].strip().split(" ")
            curr_segid = seg_text[seg_idx][0]
            window, window_idx = [], []
            text_len = len(seg_text)
            aligned_text_len = 0
            for i, char in enumerate(hyp):
                if char == eps:
                    continue

                window.append(char)
                window_idx.append(i)
                # A full segment in hyp is scanned and matched up with the VAD-segmented text
                if window == curr_seg_text:
                    ref_tokens = [ref[idx] for idx in window_idx]
                    seg_ops = [op[idx] for idx in window_idx]
                    # A hyp segment is marked as an anchor only if the current segments satisfies certain conditions
                    if satisfy_match_conditions(window, ref_tokens, seg_ops):
                        aligned_text_len += len(
                            [token for token in ref_tokens if token != eps])
                        anchor_segs.append(
                            {"segid": curr_segid, "hyp": curr_seg_text, "ref": ref_tokens, "ops": seg_ops})
                    if seg_idx < text_len - 1:
                        seg_idx += 1
                        curr_seg_text = seg_text[seg_idx][1].strip().split(" ")
                        curr_segid = seg_text[seg_idx][0]
                        window, window_idx = [], []

            uttid = Path(anchor_file).stem
            utt_dir = os.path.join(output_dir, uttid)
            mkdir_if_not_exist(utt_dir)
            ref_text_len = len([token for token in ref if token != eps])
            write_anchor_dump(anchor_segs=anchor_segs, aligned_ratio=aligned_text_len / ref_text_len,
                              output=os.path.join(utt_dir, "seg.dump"))

            # Read the timestamp information from the VAD-generated segmentation file
            with open(uttid2clip[uttid], "r") as sf:
                seg_info = json.load(
                    sf) if seg_format == "json" else parse_segments_file(sf)
            # Generate the segmentation file (in the same format as the clip_info file)
            of = os.path.join(utt_dir, f"{uttid}.json") if seg_format == "json" else os.path.join(
                utt_dir, "segments")
            write_anchor_seg(anchor_segs=anchor_segs,
                             seg_info=seg_info,
                             output=of, format=seg_format)
            write_text(anchor_segs=anchor_segs,
                       output=os.path.join(utt_dir, f"text"), eps=eps)


def main():
    parser = argparse.ArgumentParser(
        description='Determine anchor utterances based on the aligned text file.')
    parser.add_argument('--key_file', type=Path, required=True,
                        help='The full path to the key file storing the anchor/text info')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to the directory in which the segmentation files will be stored')
    parser.add_argument('--clip_info', type=Path, required=True,
                        help='The full path to the file generated by VAD that stores the timestamp for each VAD segment')
    parser.add_argument('--eps', type=str, default="<eps>",
                        help="The special epsilon token in the anchor text")
    args = parser.parse_args()

    seg_align(args.key_file, output_dir=args.output_dir,
              clip_info=args.clip_info, eps=args.eps)


if __name__ == "__main__":
    main()
