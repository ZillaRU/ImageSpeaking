import argparse
import numpy as np
import random
import os
import time
import cv2
from npuengine import EngineOV
from PIL import Image
from transformers import BertTokenizerFast
from utils.tools import *
import torch.nn as nn
from transformers import BeamSearchScorer


batch_size = 1
num_beams = 3
vocab_size = 30524
pad_token_id = 0
eos_token_id = 102
stopping_criteria_max_length = 50


class ImageSpeakingPipeline:
    def __init__(self,
        swin_path='bmodel/t2t/swin_f16.bmodel',
        tagging_head_path='bmodel/t2t/tagging_head_f16.bmodel',
        tag_encoder_path='bmodel/ram/tag_encoder.bmodel',
        tag_decoder_path='bmodel/t2t/encoder_f32.bmodel',
        tag_list='./resources/tag_list/tag2text_ori_tag_list.txt',
        label_embed_path='bmodel/t2t/label_embed.npz',
        tokenizer_path='./resources/bert-base-uncased',
        bert_path_first='bmodel/bert/ram_bert4_F16.bmodel',
        bert_cls_first_path='bmodel/bert/bert_cls_first_F16.bmodel',
        bert_cls_path='bmodel/bert/bert_cls_F16.bmodel',
        device_id=0
    ):
        self.bert_modules = [EngineOV(f'bmodel/bert/ram_bert{i}_F16.bmodel', device_id=device_id) for i in range(5, 25)]
        self.swin_infer = EngineOV(swin_path, device_id=device_id)
        self.label_embed = np.load(label_embed_path)['label_embed']
        self.tagging_head_infer = EngineOV(tagging_head_path, device_id=device_id)
        self.tag_encoder_infer = EngineOV(tag_encoder_path, device_id=device_id)
        self.tag_list = load_tag_list(tag_list)
        self.bert_infer_first = EngineOV(bert_path_first, device_id=device_id)
        self.bert_cls_infer_first = EngineOV(bert_cls_first_path, device_id=device_id)
        self.bert_cls_infer = EngineOV(bert_cls_path, device_id=device_id)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def __call__(self, image_path, length_penalty=1.0, early_stopping=False, num_return_sequences=1):
        st0 = time.time()
        ############## init ###############
        beam_scorer = BeamSearchScorer(
                            batch_size=batch_size,
                            num_beams=num_beams,
                            device='cpu',
                            length_penalty=length_penalty,
                            do_early_stopping=early_stopping,
                            num_beam_hyps_to_keep=num_return_sequences,
                        )
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device="cpu")
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        st = time.time()
        input_swin = preprocess(image_path, image_size=(384, 384))
        print("================image preprocess", time.time()-st)
        st = time.time()
        image_embed = self.swin_infer([input_swin])[0]
        print("================[TPU] swin", time.time()-st)
        image_atts = np.ones((1, 145)).astype(np.int32)
        st = time.time()
        tag = self.tagging_head_infer([self.label_embed, image_embed, image_atts])[0]
        print("================[TPU] tagging_head_infer", time.time()-st)
        tag_output = []
        index = np.argwhere(tag[0] == 1)
        token = self.tag_list[index].squeeze(axis=1)
        tag_output.append(' | '.join(token))
        image_embed = np.repeat(image_embed, num_beams, axis=0)

        tag_input_temp = []
        for tag in tag_output:
            for i in range(num_beams):
                tag_input_temp.append(tag)
                tag_output = tag_input_temp

        #_____tag encoder______
        tokenized_tag_input = self.tokenizer(tag_output, padding="max_length", return_tensors="np",max_length=40,truncation=True)
        encoder_input_ids = tokenized_tag_input.input_ids
        encoder_input_ids[:, 0] = 30523
        attention_mask = tokenized_tag_input.attention_mask.astype(np.int32)
        image_atts = np.ones((3, 145)).astype(np.int32)
        st = time.time()
        output_tagembedding = self.tag_encoder_infer([encoder_input_ids.astype(np.int32), attention_mask, image_embed.astype(np.float32), image_atts.astype(np.int32)])
        print("================[TPU] tag_encoder_infer", time.time()-st)
        last_hidden_state = output_tagembedding[0]
        
        # _____text decoder______
        # fixed input: a picture of
        input_ids_first = np.array([[30522,  1037,  3861,  1997],
                                    [30522,  1037,  3861,  1997],
                                    [30522,  1037,  3861,  1997]]).astype(np.int32)
        attention_mask = np.ones((3,4)).astype(np.int32)
        st = time.time()
        output_bert = self.bert_infer_first([input_ids_first, attention_mask, last_hidden_state])
        print("================[TPU] bert_infer_first", time.time()-st)
        sequence_output = output_bert[0]
        st = time.time()
        prediction_scores = self.bert_cls_infer_first([sequence_output])
        print("================[TPU] bert_cls_infer_first", time.time()-st)
        prediction_scores = torch.tensor(prediction_scores[0])
        input_ids = torch.tensor(input_ids_first)
        
        next_token_logits = prediction_scores[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
        next_indices = torch_int_div(next_tokens, vocab_size)
        next_tokens = next_tokens % vocab_size
        
        beam_outputs = beam_scorer.process(
                    input_ids=input_ids,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    # beam_indices=None,
                )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        input_ids_global = torch.tensor(input_ids_first)
        input_ids_global = torch.cat([input_ids_global[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        st = time.time()
        for i in range(5, 25):
            attention_mask = np.ones((3,i)).astype(np.int32)
            input_ids = input_ids_global[:, -1:]
            
            input_ids = np.array(input_ids).astype(np.int32)
            output_bert = self.bert_modules[i-5]([input_ids, attention_mask ,last_hidden_state,output_bert[1],
                                                                            output_bert[2],
                                                                            output_bert[3],
                                                                            output_bert[4],
                                                                            output_bert[5],
                                                                            output_bert[6],
                                                                            output_bert[7],
                                                                            output_bert[8],
                                                                            output_bert[9],
                                                                            output_bert[10],
                                                                            output_bert[11],
                                                                            output_bert[12],
                                                                            output_bert[13],
                                                                            output_bert[14],
                                                                            output_bert[15],
                                                                            output_bert[16],
                                                                            output_bert[17],
                                                                            output_bert[18],
                                                                            output_bert[19],
                                                                            output_bert[20],
                                                                            output_bert[21],
                                                                            output_bert[22],
                                                                            output_bert[23],
                                                                            output_bert[24]])
            sequence_output = output_bert[0]
            prediction_scores = self.bert_cls_infer([sequence_output])
            
            prediction_scores = torch.tensor(prediction_scores[0])
            input_ids_global = torch.tensor(input_ids_global)
            
            next_token_logits = prediction_scores[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    )
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                        next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                    )
            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size
            
            beam_outputs = beam_scorer.process(
                        input_ids=input_ids_global,
                        next_scores=next_token_scores,
                        next_tokens=next_tokens,
                        next_indices=next_indices,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        # beam_indices=None,
                    )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids_global = torch.cat([input_ids_global[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            if beam_scorer.is_done:
                break
        print("================[TPU] BERT modules", time.time()-st)

        sequence_outputs = beam_scorer.finalize(
                            input_ids_global,
                            beam_scores,
                            next_tokens,
                            next_indices,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            max_length=stopping_criteria_max_length,
                            # beam_indices=None,
                    )

        outputs = sequence_outputs['sequences']
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        print("================[Total]", time.time()-st0)
        return captions, tag_output[0]
