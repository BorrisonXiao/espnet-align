#!/usr/bin/env python3

import torch
import k2
from icefall.decode import (
    get_lattice,
    one_best_decoding,
)
from icefall.utils import (
    get_texts,
)

torch.set_printoptions(threshold=10_000)

word_file = "/home/cxiao7/research/espnet-cxiao/egs2/commonvoice_align/asr1/data/lang_phone/words.txt"
id2word = {}
with open(word_file, 'r') as wf:
    for line in wf.readlines():
        word, id = line.split()
        id2word[id] = word

# act ae k t
decoding_graph = k2.Fsa.from_dict(torch.load(
    "/home/cxiao7/research/espnet-cxiao/egs2/commonvoice_align/asr1/pretrain_exp/asr_train_asr_conformer5_raw_zh-HK_word_sp/graphs/HLG.pt"))

# 唔 係
nnet_output_label = [218, 218, 218, 8, 8, 0, 0, 66, 66, 66, 66, 66, 0, 0]
time_length = len(nnet_output_label)
num_channel = 323  # Number of outputs of the neural model
nnet_output = torch.ones([time_length, num_channel])
weight = 10
for index, label in enumerate(nnet_output_label):
    nnet_output[index, label] = weight
nnet_output = nnet_output.unsqueeze(0)

# nnet_output = torch.load("test_script/nnet_output.pt", map_location=torch.device('cpu'))
nnet_output_norm = nnet_output.softmax(dim=-1)
nnet_log_output = 3.5 * torch.nn.functional.log_softmax(nnet_output, dim=2)
batch_size, time_length, num_channel = nnet_log_output.shape

supervision_segments = torch.Tensor([0, 0, time_length]).to(torch.int32)
supervision_segments = supervision_segments.unsqueeze(0)
# print(supervision_segments)


lattice = get_lattice(
    nnet_output=nnet_log_output,
    decoding_graph=decoding_graph,
    supervision_segments=supervision_segments,
    search_beam=45,
    output_beam=30,
    min_active_states=7000,
    max_active_states=56000,
)
lattice_loss = lattice.get_tot_scores(log_semiring=True, use_double_scores=False)
print("====== lattice loss ========")
print(lattice_loss)

best_path = one_best_decoding(
    lattice=lattice, use_double_scores=False
)
hyps = get_texts(best_path)
print(hyps)

for hyp in hyps:
    hyp_token = [id2word[str(id)] for id in hyp]
    print(hyp_token)
