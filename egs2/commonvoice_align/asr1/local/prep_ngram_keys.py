import argparse
import os
from pathlib import Path
from utils import scpid2mid, mkdir_if_not_exist
from tqdm import tqdm


def generate_scp(input_dir, output, mscript_dir, mlevel_keys):
    """
    Generate the key file for biased ngram language model training in the format
    "<scpid> <path_to_lm_train>"
    In addition, the script also generates meeting transcript level text for
    training ngram LMs with slightly broader scope for primary decoding and
    alignment.
    """
    recorded_mids = {}
    if output:
        with open(output, "w") as f:
            with open(mlevel_keys, "w") as f2:
                for scpid in os.listdir(input_dir):
                    lm_train_fp = os.path.join(input_dir, scpid, "lm_train.txt")
                    print(scpid, lm_train_fp, file=f)

                    mid = scpid2mid(scpid)
                    mid_dir = os.path.join(mscript_dir, mid)
                    mkdir_if_not_exist(mid_dir)
                    mscp = os.path.join(mid_dir, "lm_train.txt")
                    recorded_mids[mid] = mscp
                    with open(lm_train_fp, "r") as fp:
                        with open(mscp, "a") as ofh:
                            for line in fp:
                                print(line.strip(), file=ofh)
    else:
        # Allow skipping the segment level ngram training as it is error-prone and time consuming
        with open(mlevel_keys, "w") as f:
            for scpid in tqdm(os.listdir(input_dir)):
                lm_train_fp = os.path.join(input_dir, scpid, "lm_train.txt")

                mid = scpid2mid(scpid)
                mid_dir = os.path.join(mscript_dir, mid)
                mkdir_if_not_exist(mid_dir)
                mscp = os.path.join(mid_dir, "lm_train.txt")
                recorded_mids[mid] = mscp
                with open(lm_train_fp, "r") as fp:
                    with open(mscp, "a") as ofh:
                        for line in fp:
                            print(line.strip(), file=ofh)

    with open(mlevel_keys, "w") as f:
        for mid, fp in tqdm(recorded_mids.items()):
            print(mid, fp, file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the keys for biased ngram training')
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='Input: The directory containing the script files')
    parser.add_argument('--slevel_keys', type=Path, default=None,
                        help='The generated segment level slevel_keys.scp file.')
    parser.add_argument('--mscript_dir', type=Path, required=True,
                        help='The directory to store the meeting level ngram training text for primary decoding.')
    parser.add_argument('--mlevel_keys', type=Path, required=True,
                        help='The generated meeting level mlevel_keys.scp file.')
    args = parser.parse_args()

    generate_scp(args.input_dir, args.slevel_keys, args.mscript_dir, args.mlevel_keys)


if __name__ == "__main__":
    main()
