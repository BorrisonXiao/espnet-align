#!/usr/bin/env python3
# Cihan Xiao 2022
import argparse
import logging
from pathlib import Path
import os
import k2
from utils import mkdir_if_not_exist
import torch
from icefall.lexicon import Lexicon

disambig_tokens = ["#0"]


def token_to_id(text, symbol_table, unk_id):
    assert len(text) > 0
    text_id = [symbol_table.get(
        word) if word in symbol_table else unk_id for word in text.split()]
    return text_id


def get_arc(from_state, to_state, ilabel, olabel, weight):
    return f"{from_state}\t{to_state}\t{ilabel}\t{olabel}\t{weight}"


def get_symbol_table(lang_dir):
    symbol_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
    return symbol_table


def make_g_graph_unk(text_fp: Path, output: Path, symbol_table, unk_id, disambig_ids, epsilon_id=0, weight=0, deletion_weight=0, insertion_weight=-1, draw=False):
    with open(text_fp, "r") as f:
        sentences = f.readlines()
    with open(output, "w") as f:
        arcs = []
        start_state = 0
        final_state = 1
        next_state = 2
        cur_state = start_state
        disambig_id = disambig_ids[0]

        for line in sentences:
            prev_skip_state = cur_state

            _, sentence = line.strip().split(maxsplit=1)
            word_ids = token_to_id(
                text=sentence, symbol_table=symbol_table, unk_id=unk_id)

            for token_id in word_ids:
                arc = get_arc(cur_state, next_state, token_id, token_id, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1

            arc = get_arc(prev_skip_state, cur_state, disambig_id,
                          epsilon_id, deletion_weight)
            arcs.append(arc)

        # Add the unk self-loops at start and end
        insertion_arc = get_arc(
            start_state, start_state, unk_id, unk_id, insertion_weight
        )
        arcs.append(insertion_arc)

        insertion_arc = get_arc(
            cur_state, cur_state, unk_id, unk_id, insertion_weight
        )
        arcs.append(insertion_arc)

        auxiliary_arc = get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)

        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            f.write(arc + "\n")
        f.write(final_arc)

    with open(output, "r") as f:
        fsa = k2.Fsa.from_openfst(f.read(), acceptor=False)
        # TODO: Remove
        if draw:
            fsa.labels_sym = symbol_table
            fsa.draw('/home/cxiao7/research/speech2text/random/fsa_symbols.svg',
                     title='An FSA with symbol table')
        return fsa


def make_g_graph(text_fp: Path, output: Path, symbol_table, unk_id, disambig_ids, epsilon_id=0, weight=0, deletion_weight=0, draw=False):
    with open(text_fp, "r") as f:
        sentences = f.readlines()
    with open(output, "w") as f:
        arcs = []

        start_state = 0
        final_state = 1
        next_state = 2

        disambig_id = disambig_ids[0]
        auxiliary_arc = get_arc(start_state, next_state,
                                disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        cur_state = next_state
        next_state += 1

        for line in sentences[:-1]:
            prev_skip_state = cur_state

            _, sentence = line.strip().split(maxsplit=1)
            word_ids = token_to_id(
                text=sentence, symbol_table=symbol_table, unk_id=unk_id)

            for token_id in word_ids:
                arc = get_arc(cur_state, next_state, token_id, token_id, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1

            arc = get_arc(prev_skip_state, cur_state, disambig_id,
                          epsilon_id, deletion_weight)
            arcs.append(arc)
            arc = get_arc(cur_state, next_state, disambig_id, epsilon_id, weight)
            arcs.append(arc)
            cur_state = next_state
            next_state += 1

        prev_skip_state = cur_state

        line = sentences[-1]
        _, sentence = line.strip().split(maxsplit=1)
        word_ids = token_to_id(
            text=sentence, symbol_table=symbol_table, unk_id=unk_id)

        for token_id in word_ids:
            arc = get_arc(cur_state, next_state, token_id, token_id, weight)
            arcs.append(arc)
            cur_state = next_state
            next_state += 1

        arc = get_arc(prev_skip_state, cur_state, disambig_id, epsilon_id,
                      deletion_weight)
        arcs.append(arc)

        auxiliary_arc = get_arc(cur_state, final_state, disambig_id, epsilon_id,
                                weight)
        arcs.append(auxiliary_arc)

        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            f.write(arc + "\n")
        f.write(final_arc)

    with open(output, "r") as f:
        fsa = k2.Fsa.from_openfst(f.read(), acceptor=False)
        # TODO: Remove
        if draw:
            fsa.labels_sym = symbol_table
            fsa.draw('/home/cxiao7/research/speech2text/random/fsa_symbols.svg',
                     title='An FSA with symbol table')
        return fsa


def compile_HLG(H, L, G, first_token_disambig_id, first_word_disambig_id, determinize=True, remove_epsilon=True):
    L = k2.arc_sort(L)
    #        G.labels[G.labels >= first_word_disambig_id] = 0
    #        G.__dict__["_properties"] = None
    G = k2.connect(G)
    G = k2.determinize(G)
    G = k2.arc_sort(G)
    G.lm_scores = G.scores.clone()
    LG = k2.compose(L, G)
    LG = k2.connect(LG)

    # determinize LG and remove disambig symbols
    if determinize:
        LG = k2.determinize(LG)
        LG = k2.connect(LG)

    LG.labels[LG.labels >= first_token_disambig_id] = 0
    # See https://github.com/k3-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None
    #        assert isinstance(LG.aux_labels, k2.RaggedTensor)
    #        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    if remove_epsilon:
        LG = k2.remove_epsilon(LG)
        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    LG = k2.arc_sort(LG)
    HLG = k2.compose(H, LG, inner_labels='phones')
    HLG = k2.connect(HLG)
    HLG = k2.arc_sort(HLG)

    return HLG


def build_fa_graphs(key_file, output_dir, weight, deletion_weight, lang_dir, determinize=True, remove_epsilon=True, insertion_weight=0., allow_unk=False):
    symbol_table = get_symbol_table(lang_dir)
    epsilon_id = symbol_table.get("<eps>")
    unk_id = symbol_table.get("<unk>")
    disambig_ids = []
    for token in disambig_tokens:
        disambig_ids.append(symbol_table.get(token))

    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))
    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    with open(key_file, "r") as f:
        # HLG_list = []
        for line in f:
            segid, text_fp = line.strip().split(maxsplit=1)
            if segid != "pure":
                draw = False
            else:
                draw = True
            seg_out_dir = os.path.join(output_dir, segid)
            mkdir_if_not_exist(seg_out_dir)
            G_out = os.path.join(seg_out_dir, "G.fst.txt")
            G = make_g_graph(text_fp=text_fp, epsilon_id=epsilon_id,
                             weight=weight, symbol_table=symbol_table, unk_id=unk_id, disambig_ids=disambig_ids, deletion_weight=deletion_weight, output=G_out, draw=draw) if not allow_unk else make_g_graph_unk(text_fp=text_fp, epsilon_id=epsilon_id,
                                                                                                                                                                                                                  weight=weight, symbol_table=symbol_table, unk_id=unk_id, disambig_ids=disambig_ids, deletion_weight=deletion_weight, insertion_weight=insertion_weight, output=G_out, draw=draw)
            HLG = compile_HLG(H=H, L=L, G=G, first_token_disambig_id=first_token_disambig_id,
                              first_word_disambig_id=first_word_disambig_id, determinize=determinize, remove_epsilon=remove_epsilon)
            torch.save(HLG.as_dict(), os.path.join(seg_out_dir, "HLG.pt"))
            # HLG_list.append(HLG)

        # HLGs = k2.create_fsa_vec(HLG_list)
        # torch.save(HLGs.as_dict(), output_dir / "HLG.pt")

    print(
        f"Finish compling HLG (determinize: {determinize}, remove_epsilon: {remove_epsilon})")


def main():
    """
    Build and compile the HLG graphs for flexible alignment.
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Build flexible alignment graphs.')
    parser.add_argument('--key_file', type=Path, required=True,
                        help='The full path to the key file (to the text).')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='The output directory.')
    parser.add_argument('-w', '--weight', type=float, default=0,
                        help='The weight for the linear FSA.')
    parser.add_argument('-d', '--deletion_weight', type=float, default=0,
                        help='The penalty for skipping a sentence.')
    parser.add_argument('--determinize', action="store_true",
                        help='Whether to determinize in the HLG graph compilation.')
    parser.add_argument('--allow_unk', action="store_true",
                        help='Whether to allow the insertion of unks at the start and end.')
    parser.add_argument('--insertion_weight', type=float, default=0.,
                        help='The weight (penalty) for inserting unks at the start/end.')
    parser.add_argument('--lang_dir', type=Path, required=True,
                        help='The language directory.')
    args = parser.parse_args()
    build_fa_graphs(key_file=args.key_file,
                    output_dir=args.output_dir, weight=args.weight, deletion_weight=args.deletion_weight, lang_dir=args.lang_dir, determinize=args.determinize, allow_unk=args.allow_unk, insertion_weight=args.insertion_weight, remove_epsilon=True)


if __name__ == "__main__":
    main()
