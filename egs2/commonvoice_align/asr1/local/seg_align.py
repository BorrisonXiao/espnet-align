import argparse
import os
from pathlib import Path
import re
from utils import mkdir_if_not_exist, read_anchor_file


def read_seg_text(file):
    """
    Read the text files.
    @return: A list of (segid, text) tuples.
    """
    res = []
    with open(file, "r") as f:
        for line in f:
            segid, text = line.strip().split(" ", maxsplit=1)
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


def seg_align(key_file, output_dir, clip_info, eps):
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
                # A full segment in hyp is scanned
                if window == curr_seg_text:
                    ref_tokens = [ref[idx] for idx in window_idx]
                    seg_ops = [op[idx] for idx in window_idx]
                    # Mark as anchor only if the current segments satisfies certain conditions
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
