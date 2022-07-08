import argparse


def format_sent(input_filename, output_filename, word_seg=False):
    ofh = open(output_filename, 'w', encoding='utf-8')
    with open(input_filename, 'r', encoding='utf-8') as fhd:
        uttid = 1
        for line in fhd:
            if line:
                if word_seg:
                    raise NotImplementedError
                else:
                    print(f"utt{uttid} {line.strip()}", file=ofh)
                    uttid += 1
    ofh.close()


def main():
    parser = argparse.ArgumentParser(
        description='Add utt# format for the alignment script.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input: processed, plain text file without utt#')
    parser.add_argument('--output', type=str, required=True,
                        help='Output: processed, plain text file with utt#')
    parser.add_argument('--word_seg', action='store_true',
                        help='If option provided, perform the word segmentaion (for Cantonese)')
    args = parser.parse_args()

    format_sent(args.input, args.output, word_seg=args.word_seg)


if __name__ == "__main__":
    main()
