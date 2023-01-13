import argparse
import os
from pathlib import Path
import re
from utils import read_anchor_file, mkdir_if_not_exist

SEP = "<sep>"
EPS = "***"
sep_re = re.compile(r"<sep>\d*")


def text_len(text):
    # Count the number of characters in text
    # text: list of characters
    # return: the number of characters in text
    length = 0
    for char in text:
        if char != EPS:
            length += 1
    return length


def longest_ele(arr, elem):
    # Find the longest consecutive elements in arr that are equal to elem
    # arr: list of elements
    # elem: the element to be searched
    # return: the length of the longest consecutive elements in arr that are equal to elem
    max_len = 0
    cur_len = 0
    for i in range(len(arr)):
        if arr[i] == elem:
            cur_len += 1
        else:
            if cur_len > max_len:
                max_len = cur_len
            cur_len = 0
    if cur_len > max_len:
        max_len = cur_len
    return max_len


def longest_c(op, sidx, eidx, ref, hyp, score_threshold=0.4, count_threshold=10, beam=3, verbose=False):
    # Find each segment in the reference text, compute the number of "C" in each segment
    # Using the number of "C" as the score, find the combination of segments with the highest score
    # op: list of "C", "S", "I", "D"
    # sidx: the rough starting index of the reference segment
    # eidx: the rough ending index of the reference segment
    # ref: list of characters in the reference text
    # hyp: list of characters in the hypothesis text
    # score_threshold: the threshold of the ratio of "C" in an anchor area
    # count_threshold: the threshold of the number of "C" in an anchor area
    # return: the areas of the reference text with the highest score with beam size

    # Find each segment in the reference text
    ref_seg = []
    cur_sidx = sidx
    cur_eidx = sidx + 1
    while cur_eidx < eidx:
        if is_sep(ref[cur_eidx]):
            ref_seg.append((cur_sidx, cur_eidx))
            cur_sidx = cur_eidx + 1
        cur_eidx += 1

    # Compute the number of "C" in each segment
    ref_seg_c = []
    for seg in ref_seg:
        ref_seg_c.append(op[seg[0]: seg[1]].count("C"))

    # Find the combination of segments with the highest score
    # The score is the sum of the number of "C" in each segment
    # The combination of segments must be consecutive

    # Initialize the beam
    max_score = []
    for i in range(beam):
        max_score.append((0, ()))   # (score, (start_idx, end_idx))

    i = 0
    prev_score = 0
    while i < len(ref_seg):
        j = i + 1
        while j < len(ref_seg):
            total_len = ref_seg[j][1] - ref_seg[i][0]
            # The combination of segments must be consecutive and disjoint
            score = sum(ref_seg_c[i: j + 1]) / total_len

            # Find the longest combination of segments with score > threshold
            if score < score_threshold:
                for k in range(beam):
                    if score > max_score[k][0]:
                        max_score[k] = (prev_score, (i, j))
                        max_score.sort(key=lambda x: x[0], reverse=True)
                        break
                i = j
                break
            j += 1
            prev_score = score
        i += 1

    # Post filtering, filter out all the segments with the number of max consecutive "C" < count_threshold
    for i in range(beam):
        if max_score[i][0] > 0:
            s = ref_seg[max_score[i][1][0]][0]
            e = ref_seg[max_score[i][1][1]][1]
            if e - s > count_threshold and longest_ele(op[s: e], "C") < count_threshold:
                max_score[i] = (0, ())

    if verbose:
        for i in range(beam):
            if max_score[i][0] > 0:
                s = ref_seg[max_score[i][1][0]][0]
                e = ref_seg[max_score[i][1][1]][1]
                print(ref[s: e])
                print(hyp[s: e])

    # Find the starting and ending index of the reference segment
    min_sidx = float("inf")
    max_eidx = 0
    for i in range(beam):
        if max_score[i][0] > 0:
            s = ref_seg[max_score[i][1][0]][0]
            e = ref_seg[max_score[i][1][1]][1]
            if s < min_sidx:
                min_sidx = s
            if e > max_eidx:
                max_eidx = e
    return max_score, (min_sidx, max_eidx)


def is_sep(char):
    # Check if a character is a separator
    # char: the character to be checked
    # return: True if the character is a separator, False otherwise
    if sep_re.match(char):
        return True
    return False


def sep2segid(sep):
    # Convert a separator to its corresponding segment id, that is extract the number after the SEP
    # sep: the separator to be converted
    # return: the segment id
    return int(re.findall(r'\d+', sep)[0])


def anchor(ref, hyp, op, ratio_threshold=0.2, score_threshold=0.4):
    res_text = []
    res_segid = []
    hyp_sidx, hyp_eidx = 0, 0
    prev_clean = False
    ref_eidx = 0

    # Keep track of the reference text segment id
    segid = 0
    for i, char in enumerate(ref):
        if char == SEP:
            ref[i] = f"{SEP}{segid}"
            segid += 1
    # print(ref)

    # To fix the <sep> not aligned to the start bug
    to_start = False
    for i, char in enumerate(hyp):
        if not to_start and is_sep(char):
            to_start = True
            continue
        if not to_start:
            continue
        
        if is_sep(char) and i > 0:
            hyp_sidx = hyp_eidx
            hyp_eidx = i

            # Find the corresponding reference segment's start and end index
            # If previous match is clean, the starting point in the reference text is the end point of the previous match
            if prev_clean and ref_eidx and ref_eidx < hyp_eidx:
                ref_sidx = ref_eidx + 1
            else:
                ref_sidx = hyp_sidx
            ref_eidx = hyp_eidx
            while ref_sidx > 0 and not is_sep(ref[ref_sidx - 1]):
                ref_sidx -= 1
            while ref_eidx < len(ref) and not is_sep(ref[ref_eidx]):
                ref_eidx += 1

            # print("Before", hyp_sidx, hyp_eidx, "|", ref_sidx, ref_eidx)

            # WER > threshold
            if op[hyp_sidx: hyp_eidx].count("C") / len(op[hyp_sidx: hyp_eidx]) < ratio_threshold:
                prev_clean = False
                areas, (ref_sidx, ref_eidx) = longest_c(
                    op, ref_sidx, ref_eidx, ref, hyp, score_threshold=score_threshold)
                # print(areas)

                # Drop the unmatched ones
                if areas[0][0] == 0:
                    res_segid.append(None)
                    res_text.append(None)
                    continue
            else:
                prev_clean = True

            # It appears that this is relaxed enough that it covers more than the corresponding reference segment
            # print("Hyplen:", len(clean_text(hyp[hyp_sidx: hyp_eidx]).split(" ")))
            # print("After:", hyp_sidx, hyp_eidx, "|", ref_sidx, ref_eidx)
            start_segid = sep2segid(ref[ref_sidx - 1])
            end_segid = sep2segid(ref[ref_eidx]) - 1
            res_segid.append((start_segid, end_segid))
            res_text.append(ref[ref_sidx: ref_eidx])

    return res_text, res_segid


def clean_text(text_arr):
    # Remove the SEP token and the EPS token in the text array and return the cleaned text
    return " ".join([text for text in text_arr if not is_sep(text) and text != EPS])


def scp_align(input_dir, output_dir, ratio_threshold=0.2, score_threshold=0.4):
    dumpdir = os.path.join(output_dir, "dump")
    txt_dir = os.path.join(dumpdir, "txt")
    idx_dir = os.path.join(dumpdir, "idx")
    mkdir_if_not_exist(txt_dir)
    mkdir_if_not_exist(idx_dir)

    for anchor_fname in os.listdir(input_dir):
        anchor_file = os.path.join(input_dir, anchor_fname)
        fname, ref, hyp, op, csid = read_anchor_file(anchor_file, return_uttid=False)
        alignments, alignments_idx = anchor(ref, hyp, op, ratio_threshold=ratio_threshold,
                                            score_threshold=score_threshold)

        output = os.path.join(txt_dir, fname + ".aligned")
        # The .idx file stores the index mapping
        output_idx = os.path.join(idx_dir, fname + ".idx")
        with open(output, "w") as f:
            with open(output_idx, "w") as ofh:
                for alignment, idx in zip(alignments, alignments_idx):
                    if alignment is not None:
                        print(clean_text(alignment), file=f)
                        print(idx[0], idx[1], file=ofh)
                    else:
                        print(EPS, file=f)
                        print(EPS, file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Build search map.')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='The full path to input directory')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The full path to output directory')
    parser.add_argument('--ratio_threshold', type=float, default=0.2,
                        help='The threshold for anchoring')
    parser.add_argument('--score_threshold', type=float, default=0.4,
                        help='The threshold for scoring')
    args = parser.parse_args()

    scp_align(input_dir=args.input_dir, output_dir=args.output_dir,
              ratio_threshold=args.ratio_threshold)


if __name__ == "__main__":
    main()
