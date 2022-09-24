from utils import Integerizer
import argparse
import os
from pathlib import Path
from utils import mkdir_if_not_exist, scpid2mid, extract_mid
import pinyin_jyutping_sentence
import cn2an
import re
from seg2text import text_char_seg


def eliminate_tone(scp_phoneme):
    """
    Remove the trailing numbers representing tone information.
    e.g. "ji5" => "ji"
    """
    return re.sub(r"([a-z])\d", r"\1", scp_phoneme)


def prep_txt(decoded_dir, text_map, output_dir, use_phoneme=False, ignore_tone=False, heuristic=False):
    dump_dir = os.path.join(output_dir, "dump")
    # Read the ground-truth text_map
    scps = {}
    with open(text_map, "r") as f:
        for line in f:
            scpid, scp_path = line.strip().split(" ", maxsplit=1)
            scps[scpid] = scp_path

    # Normalize and preprocess text for the align-text script
    for scpid, scp_fp in scps.items():
        mid_dir = os.path.join(dump_dir, scpid2mid(scpid))
        with open(scp_fp, "r") as f:
            scp = f.read()
            # Merge instances such as "1980 年" to "1980年" for better an2cn conversion
            scp = re.sub(r"(\d\d\d\d)\s年", r"\1年", scp)
            # Convert the numbers into characters and insert the spaces properly
            scp = text_char_seg(cn2an.transform(scp, "an2cn"))

        if use_phoneme:
            scp_phoneme = pinyin_jyutping_sentence.jyutping(
                scp, tone_numbers=True).replace("   ", " ")

            if ignore_tone:
                scp_phoneme = eliminate_tone(scp_phoneme)

            phoneme_dir = os.path.join(mid_dir, "ref", "phoneme")
            mkdir_if_not_exist(phoneme_dir)
            with open(os.path.join(phoneme_dir, scpid + ".ref"), "w") as f:
                print(f"_ {scp_phoneme}", file=f)

        char_dir = os.path.join(mid_dir, "ref", "char")
        mkdir_if_not_exist(char_dir)
        with open(os.path.join(char_dir, scpid + ".ref"), "w") as f:
            print(f"_ {scp}", file=f)

    for decoded in os.listdir(decoded_dir):
        uttid = Path(decoded).stem
        mid_dir = os.path.join(dump_dir, extract_mid(uttid))

        with open(os.path.join(decoded_dir, decoded), "r") as f:
            decoded_txt = f.read()

        if use_phoneme:
            decoded_txt_phoneme = pinyin_jyutping_sentence.jyutping(
                decoded_txt, tone_numbers=True).replace("   ", " ")

            if ignore_tone:
                decoded_txt_phoneme = eliminate_tone(decoded_txt_phoneme)

            phoneme_dir = os.path.join(mid_dir, "hyp", "phoneme")
            mkdir_if_not_exist(phoneme_dir)
            with open(os.path.join(phoneme_dir, uttid + ".hyp"), "w") as f:
                print(f"_ {decoded_txt_phoneme}", file=f)

        char_dir = os.path.join(mid_dir, "hyp", "char")
        mkdir_if_not_exist(char_dir)
        with open(os.path.join(char_dir, uttid + ".hyp"), "w") as f:
            print(f"_ {decoded_txt}", file=f)

    stats_dir = os.path.join(output_dir, "stats")
    mkdir_if_not_exist(stats_dir)
    no_map_hyp = open(os.path.join(stats_dir, "unsearched_hyp"), "w")
    token_type = "phoneme" if use_phoneme else "char"
    ref_has_map = set()
    all_refs = set()
    all_hyps = open(os.path.join(stats_dir, "hyps"), "w")
    # Create the search_map file for parallel execution
    with open(os.path.join(output_dir, "search_map"), "w") as f:
        for mid in os.listdir(dump_dir):
            hyp_dir = os.path.join(dump_dir, mid, "hyp", token_type)
            ref_dir = os.path.join(dump_dir, mid, "ref", token_type)
            for ref in os.listdir(ref_dir):
                all_refs.add(ref.rstrip(".ref"))
            for hyp in os.listdir(hyp_dir):
                print(hyp.rstrip(".hyp"), file=all_hyps)
                mapped = False
                for ref in os.listdir(ref_dir):
                    if heuristic:
                        hyp_spkid = hyp.strip().split("_")[0]
                        ref_spkid = ref.strip().split("_")[0]
                        if ref_spkid in hyp_spkid:
                            print(os.path.join(hyp_dir, hyp),
                                  os.path.join(ref_dir, ref), file=f)
                            mapped = True
                            ref_has_map.add(ref.rstrip(".ref"))
                    else:
                        print(os.path.join(hyp_dir, hyp),
                              os.path.join(ref_dir, ref), file=f)
                        mapped = True
                        ref_has_map.add(ref.rstrip(".ref"))
                if not mapped:
                    print(hyp.rstrip(".hyp"), file=no_map_hyp)
    with open(os.path.join(stats_dir, "unsearched_ref"), "w") as f:
        for ref in all_refs - ref_has_map:
            print(ref, file=f)
    with open(os.path.join(stats_dir, "refs"), "w") as f:
        for ref in all_refs:
            print(ref, file=f)
    no_map_hyp.close()
    all_hyps.close()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the text for Levenshtein distance computation.')
    parser.add_argument('--decoded_dir', type=str, required=True,
                        help='The full path to the directory in which decoded text files are stored')
    parser.add_argument('--text_map', type=str, required=True,
                        help='The full path to the text_map file generated for the ground-truth text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The full path to the directory in which the comparison file will be stored')
    parser.add_argument('--heuristic', action='store_true',
                        help='If option provided, search space of each hyp script will be limited to ref script that contains the spkid')
    parser.add_argument('--use_phoneme', action='store_true',
                        help='If option provided, the text will be converted to phoneme using pinyin_jyutping_sentence')
    parser.add_argument('--ignore_tone', action='store_true',
                        help='If option provided, the converted phonemes will disregard tones')
    args = parser.parse_args()

    prep_txt(args.decoded_dir, args.text_map, args.output_dir,
             args.use_phoneme, args.ignore_tone, args.heuristic)


if __name__ == "__main__":
    main()
