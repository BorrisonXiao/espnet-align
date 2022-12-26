import argparse
import os
from pathlib import Path
from utils import mkdir_if_not_exist, scpid2mid, extract_mid
import pinyin_jyutping_sentence
import re


def eliminate_tone(scp_phoneme):
    """
    Remove the trailing numbers representing tone information.
    e.g. "ji5" => "ji"
    """
    return re.sub(r"([a-z])\d", r"\1", scp_phoneme)


def prep_ref(text_map, output_dir, use_phoneme=False, ignore_tone=False):
    dump_dir = os.path.join(output_dir, "dump")
    # Read the ground-truth text_map
    scps = {}
    with open(text_map, "r") as f:
        for line in f:
            scpid, scp_path = line.strip().split(" ", maxsplit=1)
            scps[scpid] = scp_path

    # The following operation relies on the fact that the text_map file is sorted
    # by mid and scp sequence id
    prev_mid = None
    cache = []

    # Combine reference text with <sep> as the delimiter indicating file switch
    for i, (scpid, scp_fp) in enumerate(scps.items()):
        mid = scpid2mid(scpid)

        if mid != "M19120002" and mid != "M19120036":
            continue

        if not prev_mid:
            prev_mid = mid
        
        with open(scp_fp, "r") as f:
            scp = f.read().strip()

        # If new mid is reached, close the previous file pointer and open a new one
        if prev_mid != mid:
            prev_mid_dir = os.path.join(dump_dir, prev_mid)
            if use_phoneme:
                phoneme_dir = os.path.join(prev_mid_dir, "ref", "phoneme")
                mkdir_if_not_exist(phoneme_dir)
                cache_ph = [pinyin_jyutping_sentence.jyutping(s, tone_numbers=True).replace("   ", " ") for s in cache]
                cache_ph = [eliminate_tone(s) for s in cache_ph] if ignore_tone else cache_ph
                with open(os.path.join(phoneme_dir, prev_mid + ".ref"), "w") as ofh:
                    print("_", "<sep>", " <sep> ".join(cache_ph), "<sep>", file=ofh)

            char_dir = os.path.join(prev_mid_dir, "ref", "char")
            mkdir_if_not_exist(char_dir)
            with open(os.path.join(char_dir, prev_mid + ".ref"), "w") as ofh:
                print("_", "<sep>", " <sep> ".join(cache), "<sep>", file=ofh)
            cache = []
            prev_mid = mid

        cache.append(scp)

        if i == len(scps) - 1:
            mid_dir = os.path.join(dump_dir, mid)
            if use_phoneme:
                phoneme_dir = os.path.join(mid_dir, "ref", "phoneme")
                mkdir_if_not_exist(phoneme_dir)
                cache_ph = [pinyin_jyutping_sentence.jyutping(s, tone_numbers=True).replace("   ", " ") for s in cache]
                cache_ph = [eliminate_tone(s) for s in cache_ph] if ignore_tone else cache_ph
                with open(os.path.join(phoneme_dir, mid + ".ref"), "w") as ofh:
                    print("_", "<sep>", " <sep> ".join(cache_ph), "<sep>", file=ofh)

            char_dir = os.path.join(mid_dir, "ref", "char")
            mkdir_if_not_exist(char_dir)
            with open(os.path.join(char_dir, mid + ".ref"), "w") as ofh:
                print("_", "<sep>", " <sep> ".join(cache), "<sep>", file=ofh)


def prep_hyp(text_map, output_dir, use_phoneme=False, ignore_tone=False):
    dump_dir = os.path.join(output_dir, "dump")
    # Read the ground-truth text_map
    hyps = {}
    with open(text_map, "r") as f:
        for line in f:
            uttid, hyp_fp = line.strip().split(" ", maxsplit=1)
            hyps[uttid] = hyp_fp

    # The following operation relies on the fact that the text_map file is sorted
    # by mid and scp sequence id
    prev_mid = None
    cache = []

    # Combine reference text with <sep> as the delimiter indicating file switch
    for i, (uttid, hyp_fp) in enumerate(hyps.items()):
        mid = extract_mid(uttid)

        if mid != "M19120002" and mid != "M19120036":
            continue

        if prev_mid == None:
            prev_mid = mid
        
        with open(hyp_fp, "r") as f:
            hyp = f.read().strip()

        # If new mid is reached, close the previous file pointer and open a new one
        if prev_mid != mid:
            prev_mid_dir = os.path.join(dump_dir, prev_mid)
            if use_phoneme:
                phoneme_dir = os.path.join(prev_mid_dir, "hyp", "phoneme")
                mkdir_if_not_exist(phoneme_dir)
                cache_ph = [pinyin_jyutping_sentence.jyutping(s, tone_numbers=True).replace("   ", " ") for s in cache]
                cache_ph = [eliminate_tone(s) for s in cache_ph] if ignore_tone else cache_ph
                with open(os.path.join(phoneme_dir, prev_mid + ".hyp"), "w") as ofh:
                    print("_", "<sep>", " <sep> ".join(cache_ph), "<sep>", file=ofh)

            char_dir = os.path.join(prev_mid_dir, "hyp", "char")
            mkdir_if_not_exist(char_dir)
            with open(os.path.join(char_dir, prev_mid + ".hyp"), "w") as ofh:
                print("_", "<sep>", " <sep> ".join(cache), "<sep>", file=ofh)
            cache = []
            prev_mid = mid

        cache.append(hyp)

        if i == len(hyps) - 1:
            mid_dir = os.path.join(dump_dir, mid)
            if use_phoneme:
                phoneme_dir = os.path.join(mid_dir, "hyp", "phoneme")
                mkdir_if_not_exist(phoneme_dir)
                cache_ph = [pinyin_jyutping_sentence.jyutping(s, tone_numbers=True).replace("   ", " ") for s in cache]
                cache_ph = [eliminate_tone(s) for s in cache_ph] if ignore_tone else cache_ph
                with open(os.path.join(phoneme_dir, mid + ".hyp"), "w") as ofh:
                    print("_", "<sep>", " <sep> ".join(cache_ph), "<sep>", file=ofh)

            char_dir = os.path.join(mid_dir, "hyp", "char")
            mkdir_if_not_exist(char_dir)
            with open(os.path.join(char_dir, mid + ".hyp"), "w") as ofh:
                print("_", "<sep>", " <sep> ".join(cache), "<sep>", file=ofh)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the text for Levenshtein distance computation.')
    parser.add_argument('--text_map', type=str, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which the comparison file will be stored')
    parser.add_argument('--use_phoneme', action='store_true',
                        help='If option provided, the text will be converted to phoneme using pinyin_jyutping_sentence')
    parser.add_argument('--ignore_tone', action='store_true',
                        help='If option provided, the converted phonemes will disregard tones')
    parser.add_argument('--ref', action='store_true',
                        help='If option provided, the script will run on reference text mode, otherwise hyp')
    args = parser.parse_args()

    ref_mode = args.ref
    if ref_mode:
        prep_ref(args.text_map, args.output_dir, args.use_phoneme, args.ignore_tone)
    else:
        prep_hyp(args.text_map, args.output_dir, args.use_phoneme, args.ignore_tone)


if __name__ == "__main__":
    main()
