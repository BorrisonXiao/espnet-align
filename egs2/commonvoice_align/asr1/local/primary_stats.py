import argparse
import os
from pathlib import Path
from utils import read_pdump_file


def compute_stats(dump, groud_truth, stats_dir):
    """
    Compute the stats of the primary alignment results.
    """
    truth = {}
    dubious_truth = {}
    clear_truth = {}
    unknown_truth = set()
    # Read the ground truth file, e.g. speech2text/align_data_v0/processed/metadata/metadata_map.txt
    with open(groud_truth, "r") as f:
        for line in f:
            line = line.strip()
            ref, hyp = line.split()
            refid = ref
            if line.endswith("***"):
                refid = refid.rstrip("_*")
                unknown_truth.add(refid)
                truth[refid] = "unknown"
                continue
            if ref.endswith("_*"):
                refid = ref.rstrip("_*")
                dubious_truth[refid] = hyp
            else:
                clear_truth[refid] = hyp
            truth[refid] = hyp

    dumpinfo = read_pdump_file(dump)

    # Compute accuracy on different sets
    correct, d_correct, c_correct = 0, 0, 0
    errors, d_errors, c_errors = [], [], []
    d_len, c_len = 0, 0
    unkowns = []
    matched_count = 0
    for ref, info in dumpinfo.items():
        # Skip data without ground-truth
        if ref not in truth:
            continue
        if truth[ref] == info['hyp']:
            correct += 1
        else:
            if truth[ref] != "unknown":
                errors.append((ref, info['hyp'], truth[ref]))
            else:
                unkowns.append((ref, *(info.values())))
                continue
        matched_count += 1

        if ref in dubious_truth:
            d_len += 1
            if dubious_truth[ref] == info['hyp']:
                d_correct += 1
            else:
                d_errors.append((ref, info['hyp'], dubious_truth[ref]))

        if ref in clear_truth:
            c_len += 1
            if clear_truth[ref] == info['hyp']:
                c_correct += 1
            else:
                c_errors.append((ref, info['hyp'], clear_truth[ref]))

    # Compute search coverage, i.e. the number of segments are in the search space
    with open(os.path.join(stats_dir, "refs"), "r") as f:
        all_refs = len(f.readlines())
    with open(os.path.join(stats_dir, "hyps"), "r") as f:
        all_hyps = len(f.readlines())
    with open(os.path.join(stats_dir, "searched_ref"), "r") as f:
        searched_refs = len(f.readlines())
    with open(os.path.join(stats_dir, "searched_hyp"), "r") as f:
        searched_hyps = len(f.readlines())
    with open(os.path.join(stats_dir, "unmatched_ref"), "r") as f:
        unmatched_refs = len(f.readlines())
    with open(os.path.join(stats_dir, "unmatched_hyp"), "r") as f:
        unmatched_hyps = len(f.readlines())
    with open(os.path.join(stats_dir, "summary"), "w") as f:
        print(f"Script Search Coverage: {(searched_refs / all_refs):.2f}", file=f)
        print(f"Audio Search Coverage: {(searched_hyps / all_hyps):.2f}", file=f)
        print(f"Script Match Rate: {(1 - unmatched_refs / searched_refs):.2f}", file=f)
        print(f"Audio Match Rate: {(1 - unmatched_hyps / searched_hyps):.2f}", file=f)
        print("Overall Matching Acc:", f"{correct / matched_count:.2f}", file=f)
        print("Errors", file=f)
        for err in errors:
            print(f"\t{err}", file=f)
        print("Dubious Set Matching Acc:", f"{d_correct / d_len:.2f}", file=f)
        print("Errors", file=f)
        for err in d_errors:
            print(f"\t{err}", file=f)
        print("Clear Set Matching Acc:", f"{c_correct / c_len:.2f}", file=f)
        print("Errors", file=f)
        for err in c_errors:
            print(f"\t{err}", file=f)
        print("Results for the UNK", file=f)
        for unk in unkowns:
            print(f"\t{unk}", file=f)

    # Compute dubious set accuracy
    d_correct = 0
    d_errors = []
    for ref, info in dumpinfo.items():
        if ref not in truth:
            continue
        if truth[ref] == info['hyp']:
            correct += 1
        else:
            errors.append((ref, info['hyp'], truth[ref]))


def main():
    parser = argparse.ArgumentParser(
        description='Compute the stats of the primary alignment results')
    parser.add_argument('--dump', type=Path, required=True,
                        help='The alignment result dump file')
    parser.add_argument('--groud_truth', type=Path, required=True,
                        help='The full path to the file in which the groud-truth mappings are stored')
    parser.add_argument('--stats_dir', type=Path, default=None,
                        help='The full path to the directory which stores all previous stages stats')
    args = parser.parse_args()

    compute_stats(args.dump, args.groud_truth, args.stats_dir)


if __name__ == "__main__":
    main()
