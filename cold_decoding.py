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
from transformers import DistilBertModel, DistilBertTokenizer

import math 
from annoy import AnnoyIndex

from cold_args import options
from decoder import decode
from model.inductiveAttentionModel import GPT2InductiveAttentionHeadModel
from model.mese import C_UniversalCRSModel
from GPT2ForwardBackward import OpenGPT2LMHeadModel


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


def lexical_generation(red, model, model_back, tokenizer, device, args):

    data, outfile, fw, fw_pretty = red

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d["concept_set"])
        constraints = d["concept_set"].split("#")

        constraints = ' '.join(constraints)
        # x = "<|endoftext|>"
        if "starter" in d:
            x = d["starter"]
        else:
            x = '<|endoftext|>'
        z = constraints
        z_keywords = constraints

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords


        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(model, model_back, tokenizer, device, x, z, None, args,
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


def load_pretrained_model(args, device):
    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, output_hidden_states=True,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
    model.to(device)
        # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

    model_back = OpenGPT2LMHeadModel.from_pretrained(
        args.back_model, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_first_dropout=0)
    model_back.to(device)


    return model, model_back, tokenizer


def load_my_model(args, device):
    CKPT = args.pretrained_model
    # device = torch.device(0)

    bert_tokenizer = DistilBertTokenizer.from_pretrained("../../../offline_transformers/distilbert-base-uncased/tokenizer/")
    bert_model_recall = DistilBertModel.from_pretrained('../../../offline_transformers/distilbert-base-uncased/model/')
    bert_model_rerank = DistilBertModel.from_pretrained('../../../offline_transformers/distilbert-base-uncased/model/')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("../../../offline_transformers/gpt2/tokenizer/")
    gpt2_model = GPT2InductiveAttentionHeadModel.from_pretrained('../../../offline_transformers/gpt2/model/')

    
    REC_TOKEN = "[REC]"
    REC_END_TOKEN = "[REC_END]"
    SEP_TOKEN = "[SEP]"
    PLACEHOLDER_TOKEN = "[MOVIE_ID]"
    gpt_tokenizer.add_tokens([REC_TOKEN, REC_END_TOKEN, SEP_TOKEN, PLACEHOLDER_TOKEN])
    gpt2_model.resize_token_embeddings(len(gpt_tokenizer)) 


    items_db_path = args.kb_path
    items_db = torch.load(items_db_path)

    universal_model =  C_UniversalCRSModel(
        gpt2_model, 
        bert_model_recall, 
        bert_model_rerank, 
        gpt_tokenizer, 
        bert_tokenizer, 
        device, 
        items_db, 
        rec_token_str=REC_TOKEN, 
        rec_end_token_str=REC_END_TOKEN
    )
    
    ########## Loading Weights for the Model to generate ###############
    universal_model.to(device)
    universal_model.load_state_dict(torch.load(CKPT,map_location=device))

    universal_model.annoy_base_constructor()
    _ = universal_model.lm_expand_wtes_with_items_annoy_base()       

    # Freeze GPT-2 weights
    for param in universal_model.language_model.parameters():
        param.requires_grad = False   
    

    model_back = OpenGPT2LMHeadModel.from_pretrained(
        args.back_model, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_first_dropout=0)
    model_back.to(device)
    


    return universal_model.language_model, model_back, gpt_tokenizer

def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"


    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # model, tokenizer = load_pretrained_model(args,device) 
    model, model_back, tokenizer = load_my_model(args,device)
   
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False


    model_back.eval()
    # Freeze GPT-2 weights
    for param in model_back.parameters():
        param.requires_grad = False
    
    red = read_data(args)

    if "lexical" in args.mode:
        lexical_generation(red, model, model_back, tokenizer, device, args)


if __name__ == "__main__":
    main()
