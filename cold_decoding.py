#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import json
import torch

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from cold_args import options
from decoder import decode

stop_words = set(stopwords.words('english'))


def read_data(args):
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]


    outfile = '%if_zx%s_seed%d_%d_%d_%s_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
              '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.if_zx,
                  args.version,
                  args.seed,
                  args.start,
                  args.end,
                  args.mode,
                  args.constraint_weight,
                  args.abductive_c2_weight,
                  args.lr_nll_portion,
                  args.length,
                  args.topk,
                  args.num_iters,
                  args.frozen_length,
                  args.win_anneal_iters,
                  args.noise_iters,
                  args.gs_std,
                  args.stepsize,
                  args.stepsize_ratio,
                  args.stepsize_iters,
                  args.large_noise_iters,
                  args.large_gs_std)
    print("outputs: %s" % outfile)

    fw = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    return data, outfile, fw, fw_pretty


def lexical_generation(red, model, tokenizer, device, args):

    data, outfile, fw, fw_pretty = red

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d["concept_set"])
        constraints = d["concept_set"].split("#")

        constraints = ' '.join(constraints)
        x = "<|endoftext|>"
        z = constraints
        z_keywords = constraints

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords



        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(model, tokenizer, device, x, z, None, args,
                                               zz=z_keywords)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'constraints': constraints,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)
        print('Output to: \t', outfile)

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"


    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, output_hidden_states=True,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
    model.to(device)


    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

    red = read_data(args)

    if "lexical" in args.mode:
        lexical_generation(red, model, tokenizer, device, args)


if __name__ == "__main__":
    main()
