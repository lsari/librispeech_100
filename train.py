# Author: Leda Sari

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import LibriSpeechDataset
import models

# For decoding
from ctcdecode import CTCBeamDecoder
from Levenshtein import distance as levenshtein_distance


# TODO: Add some seed 

with open("train_args.json", 'r') as f:
    args = json.load(f)

ctc_loss = nn.CTCLoss(blank=args["pad_token"], reduction="none")

    
train_set = LibriSpeechDataset(args["train_set"], args)
valid_set = LibriSpeechDataset(args["valid_set"], args)

train_collate_fn = train_set.pad_collate

train_loader = DataLoader(
    train_set,
    batch_size=args["batch_size"],
    collate_fn=train_collate_fn,
    shuffle=False,
    num_workers=args["num_workers"]
)

valid_loader = DataLoader(
    valid_set,
    batch_size=args["batch_size"],
    collate_fn=train_collate_fn,
    num_workers=args["num_workers"]
)

device = torch.device('cuda' if args["use_gpu"] else "cpu")
model = models.CNNLSTM(args)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), args["lr"])


for e in range(args["num_epochs"]):
    model.train()
    for (index, features, trns, input_lengths) in (train_loader):
        features = features.float().to(device)
        features = features.transpose(1,2).unsqueeze(1)
        trns = trns.long().to(device)
        input_lengths = input_lengths.long().to(device)

        optimizer.zero_grad()
        log_y, output_lengths = model(features, input_lengths, trns)
        
        target_lengths = torch.IntTensor([
            len(y[y != args["pad_token"]]) for y in trns
        ])
        train_ctc_loss = torch.mean(ctc_loss(log_y, trns, output_lengths, target_lengths)/(target_lengths.float().to(device))) 
                
        train_ctc_loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), args["clip"])
        optimizer.step()
        print(train_ctc_loss.data)
        
    valid_ctc_loss = model.eval_loss(valid_loader, ctc_loss, device)
    print("Valid loss at epoch {}: {}".format(e, valid_ctc_loss.data))

    # TODO: Save model, update lr? 
    
    
# Test phase:
print("==========================")
print("===     TEST PHASE     ===")
print("==========================")


with open(args["labels_file"], 'r') as f:
    label_dict=  json.load(f)

rev_label_dict = {v: k for k, v in label_dict.items()}
    
decoder = CTCBeamDecoder(
    labels=[str(c) for c in rev_label_dict.keys()], beam_width=1
)

model.eval()
total_error = 0.0
total_length = 0.0
with torch.no_grad():
    for (index, features, trns, input_lengths) in (train_loader):
        features = features.float().to(device)
        features = features.transpose(1,2).unsqueeze(1)
        trns = trns.long().to(device)
        input_lengths = input_lengths.int() # long().to(device)
        
        log_y, output_lengths = model(features, input_lengths, trns)
        
        target_lengths = torch.IntTensor([
            len(y[y != args["pad_token"]]) for y in trns
        ])

        # print(log_y.size(), input_lengths.size())
        out, scores, offsets, seq_lens = decoder.decode(torch.exp(log_y).transpose(0,1).detach().cpu(), input_lengths)
        for hyp, trn, length in zip(out, trns, seq_lens): # iterate batch
            best_hyp = hyp[0,:length[0]]
            # best_hyp_str = ''.join(list(map(chr, best_hyp)))
            hh = ''.join([rev_label_dict[i.item()] for i in best_hyp])
            t = trn.detach().cpu().tolist()
            t = [ll for ll in t if ll != 0]
            tlength = len(t)
            # truth_str = ''.join(list(map(chr, t)))
            tt = ''.join([rev_label_dict[i] for i in t])
            print("Truth: ", tt)
            print("Hypothesis: ", hh)
            error = levenshtein_distance(tt, hh)
            total_error += error
            total_length += tlength
        
            
