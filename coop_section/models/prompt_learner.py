import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import math, re, functools
import numpy as np

import time, sys, os
import torch.nn as nn
sys.path.append("../")

from clip_code import clip
from clip_code.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, N_CTX, n_verb, VERB_TOKEN_POSITION, clip_model, nomal = 'nomal'):
        super().__init__()
        n_ctx = N_CTX  # numbers of words in prompts
        self.n_verb = n_verb
        print("n_verb:", self.n_verb)
        # ctx_init = opt.CTX_INIT # initial prompts
        self.verb_token_position = VERB_TOKEN_POSITION

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]


        print("Initializing verb-specific contexts")
        ctx_vectors = torch.empty(self.n_verb, n_ctx, ctx_dim, dtype = dtype)
        if nomal == 'orthogonal':
            torch.nn.init.orthogonal_(ctx_vectors)
            print("=======================================")
            print("===> ctx vectors using orthogonal_ inital.")
            print("=======================================")
        elif nomal == 'nomal':
            nn.init.normal_(ctx_vectors, std = 0.02)
        else:
            assert 0
        
        self.prompt_prefix = " ".join(["X"] * n_ctx) # index:343
        #prompt_prefix_token = clip.tokenize(self.prompt_prefix)[:, 1:n_ctx + 1] #[1, 77] -> [1, n_ctx]
        self.register_buffer("prompt_prefix_token", clip.tokenize(self.prompt_prefix)[:, 1:n_ctx + 1])

        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.token_embedding = clip_model.token_embedding #=====是否要冻结？？？
        self.n_ctx = n_ctx
    
    def forward(self, nouns_token, nouns_numbers):
        '''
        nouns_token:[batchsize, 77]
        nous_numbers:[batchsize]
        '''
        ctx = self.ctx

        ## 1. get full token
        batchsize = nouns_token.shape[0]
        ## concat token
        concat_token_list = []
        for i in range(batchsize):
            #prompt_prefix_token [1, n_ctx]
            sub_nouns_numbers = nouns_numbers[i].item()
            sub_token_prefix = nouns_token[i:i+1, :1]
            sub_token_suffix = nouns_token[i:i+1, 1+sub_nouns_numbers:]
            sub_token_nouns = nouns_token[i:i+1, 1:sub_nouns_numbers+1]
            #print(sub_token_prefix.shape, sub_token_suffix.shape, sub_token_nouns.shape, self.prompt_prefix_token.shape)
            sub_concar_token = torch.cat((sub_token_prefix, sub_token_nouns, self.prompt_prefix_token, sub_token_suffix), dim = -1)[:, :77]#[1, 77]
            concat_token_list.append(sub_concar_token)
        concat_token = torch.cat(concat_token_list, dim = 0)#[batchsize, 77]
        #print("concat_token:", concat_token.shape)


        ## 2. get full embedding
        # with torch.no_grad():
        if self.token_embedding.weight.requires_grad:
            nouns_token_embedding = self.token_embedding(nouns_token)#[batchsize, 77, 512]
        else:
            with torch.no_grad():nouns_token_embedding = self.token_embedding(nouns_token)
            #print("nouns_token_embedding:", nouns_token_embedding.shape)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_verb, -1, -1) #[n_verb, n_ctx, ctx_dim]

        prompts = []
        for i in range(batchsize):
            sub_nouns_numbers = nouns_numbers[i].item()
            sub_nouns_token_embedding = nouns_token_embedding[i, :, :] #[77, 512]
            #expand
            sub_nouns_token_embedding = sub_nouns_token_embedding.unsqueeze(0).expand(self.n_verb, -1, -1)#[n_verb, 77, 512]
            sub_nouns_token_prefix_embedding = sub_nouns_token_embedding[:, :1, :]#[n_verb, 1, 512]
            sub_nouns_token_suffix_embedding = sub_nouns_token_embedding[:, 1+sub_nouns_numbers:, :]#[n_verb, *, 512]
            sub_nouns_token_nouns_embedding = sub_nouns_token_embedding[:, 1:sub_nouns_numbers+1, :]#[n_verb, numbers, 512]
            #print(sub_nouns_token_embedding.shape, sub_nouns_token_prefix_embedding.shape, sub_nouns_token_suffix_embedding.shape, sub_nouns_token_nouns_embedding.shape)
            prompt = torch.cat(
                [
                    sub_nouns_token_prefix_embedding,  # (n_verb, 1, dim)
                    sub_nouns_token_nouns_embedding,     # (n_verb, numbers, dim)
                    ctx, #[n_verb, n_ctx, ctx_dim]
                    sub_nouns_token_suffix_embedding,  # (n_verb, *, dim)
                ],
                dim=1,
            )[:, :77, :]# (n_verb, 77, dim)
            #print("prompt:", prompt.shape)
            prompts.append(prompt.unsqueeze(0))

        prompts = torch.cat(prompts, dim=0)# (batchsize, n_verb, 77, dim)
       # print("prompts:", prompts.shape)
    
        return prompts, concat_token#(batchsize, n_verb, 77, dim), #[batchsize, 77]

class clip_prompt_sentence_encoder(nn.Module):
    def __init__(self, opt, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(opt.N_CTX, opt.n_verb, opt.VERB_TOKEN_POSITION, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_ctx = opt.N_CTX
        self.n_verb = opt.n_verb

    def forward(self, nouns_token, nouns_numbers):

        prompts, tokenized_prompts = self.prompt_learner(nouns_token, nouns_numbers)#(batchsize, n_verb, 77, dim), #[batchsize, 77]
        batchsize = prompts.shape[0]
        prompts = prompts.reshape(-1, 77, self.ctx_dim)#(batchsize*n_verb, 77, dim)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).expand(self.n_verb, batchsize, 77).permute(1, 0, 2).reshape(self.n_verb * batchsize, 77)#(batchsize*n_verb, 77)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features ##(batchsize*n_verb, dim)