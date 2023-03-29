import torch 

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DistilBertModel, DistilBertTokenizer

from model.mese import C_UniversalCRSModel
from model.inductiveAttentionModel import GPT2InductiveAttentionHeadModel
from dataset import RecDataset
from torch.utils.data import DataLoader
import json 

fw = open('data/durecdial/durecdial2.dev.jsonl','w')

def jsonify(batch, model, tokenizer, item_id_2_lm_token_id):
        role_ids, dialogues = batch
        dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ , _ in dialogues]

        past_tokens = None
        
        for turn_num in range(len(role_ids)):
            dial_turn_inputs = dialog_tensors[turn_num]
            _, gold_recommended_ids, target_goal = dialogues[turn_num]
            
        
            gold_item_ids = []; gold_item_titles = []
            if gold_recommended_ids != None:
                for r_id in gold_recommended_ids:
                    gold_item_ids.append(item_id_2_lm_token_id[r_id])
                    title = model.items_db[r_id]
                    title = title.split('[SEP]')[0].strip()
                    gold_item_titles.append(title)
                gold_item_ids = torch.tensor([gold_item_ids])
            
            if role_ids[turn_num] == 0: # User
                if turn_num == 0:
                    past_tokens = dial_turn_inputs
                elif turn_num!= 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
            else: # System

                if gold_recommended_ids:
                    rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    if gold_recommended_ids!= None:
                        past_tokens = torch.cat((past_tokens, rec_start_token, gold_item_ids, rec_end_token), dim=1)
                    else:
                        past_tokens = past_tokens
                else:
                    past_tokens = past_tokens


                input_ids= torch.cat((past_tokens, torch.tensor([[32, 25]])),dim = 1)
                # for i, id in enumerate(input_ids[0]):
                #     if id > len(tokenizer):
                #         input_ids[0][i] = len(tokenizer)-1
                if gold_recommended_ids:
                    text = tokenizer.batch_decode(input_ids)
                    dc = {"concept_set":"[MOVIE_ID]", "starter": text[0]}

                    fw.write(json.dumps(dc)+
                            '\n')
        
                if turn_num != 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
                



if __name__ == '__main__':
    CKPT = '../KBConvRec/mese_baseline/runs/new_model_rec.pt'

    
    device = torch.device('cpu')

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


    items_db_path = '../KBConvRec/mese_baseline/data/kb/durecdial2_db'
    items_db = torch.load(items_db_path)



    model =  C_UniversalCRSModel(
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
    model.load_state_dict(torch.load(CKPT,map_location=device))

    test_path = "../KBConvRec/mese_baseline/data/processed/durecdial2/durecdial2_all_dev_placeholder_updated"
    test_dataset = RecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)
    # Data loader 
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate)

    model.annoy_base_constructor()
    item_id_2_lm_token_id = model.lm_expand_wtes_with_items_annoy_base()          

    for batch in test_dataloader:
        jsonify(batch[0], model, gpt_tokenizer, item_id_2_lm_token_id)