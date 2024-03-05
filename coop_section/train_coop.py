import os, sys
sys.path.append(r"/home/likegiver/Desktop/codes/2023_5/whole_work")
import torch.nn as nn
from torch.nn import functional as F
from  torch.cuda.amp import autocast
from dataset import MyDataset
import torch.utils.data as Data
import torch.distributed as dist
from tqdm import tqdm
import time, random, logging
import argparse
from pathlib import Path
import yaml

from models.prompt_learner import clip_prompt_sentence_encoder
from models.ImageCLIP import ImageCLIP
from models.coop import CustomCLIP
import clip_code
from clip_code import clip
from utils.base_utils import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import create_logits, create_multiverbs_logits, saving_models
import options

## 1. initialize seed, opt and device ##
setup_seed(2023)
opt_class = options.BaseOptions()
opt = opt_class.parse()
device = torch.device("cuda")

if (not opt.is_distributed) or (dist.get_rank() == 0):
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)
    if not os.path.exists(opt.modeldir):
        os.makedirs(opt.modeldir)

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.basicConfig(filename=os.path.join(opt.logdir, opt.log_name), filemode='w', level=logging.DEBUG,
                                                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    root_logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(stdout_handler)
    opt_class.print_options(opt)

## 2. loading models and set models##
backbone_name = 'ViT-B-32'
clip_model, preprocessor = clip.load('clip_code/checkpoints/ViT-B-32.pt', 
                device="cuda", jit=False, tsm=False, T=8, dropout=0.0 , emb_dropout=0.0 ,pretrain=True, joint = False) #Must set jit=False for training  ViT-B/32

### visual models ###
# model_image = ImageCLIP(clip_model)

### sentence models ###
# model_text = clip_prompt_sentence_encoder(opt, clip_model)

# model_image = model_image.to(device)
# model_text = model_text.to(device)

### frozen net ###
frozen(clip_model)
# frozen(model_image)
### frozen text embedding and postion. ###
# for name, parameter in model_text.named_parameters():
#         if name != 'module.prompt_learner.ctx':
#             parameter.requries_grad = False

## 3. initialize dataset ##
### 创建Dataset对象 ###
import torch
import pickle
# 加载.pt文件中的张量到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images_tensor = torch.load('data/image_features.pt', map_location=device)

with open('data/train_photo_ids.pkl', 'rb') as f:
    train_photo_ids = pickle.load(f)

with open('data/table_photo_ids.pkl', 'rb') as f:
    table_photo_ids = pickle.load(f)

with open('data/class_names.pkl', 'rb') as f:
    class_names_list = pickle.load(f)

classname_indexes = []
classnames = ['General Impression',  # just use the 7 classes string list 
    'Subject of Photo',
    'Composition & Perspective',
    'Use of Camera,Exposure & Speed',
    'Depth of Field',   
    'Color & Lighting',
    'Focus']
for class_name in class_names_list:
    classname_indexes.append(classnames.index(class_name))

my_dataset = MyDataset(images_tensor, classname_indexes, train_photo_ids, table_photo_ids)

### 创建DataLoader对象 ###
batch_size = opt.batchsize
train_loader = Data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

## 4. define optimizer and lr_scheduler ##
# optimizer = _optimizer(opt, model_image, model_text)
# lr_scheduler = _lr_scheduler(opt, optimizer)

# print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
# clip_model = load_clip_to_cpu(cfg)

# if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
#     # CLIP's default precision is fp16
#     clip_model.float()

print("Building custom CLIP")
coop_model = CustomCLIP(opt, classnames, clip_model) # opt is transfered into the CustomCLIP.PromptLearner directly, which uses opt.n_ctx and opt.class_token_position

print("Turning off gradients in coop_model except the prompt_learner")
for name, param in coop_model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

# if cfg.MODEL.INIT_WEIGHTS:
#     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
if opt.load_pretrained:
    checkpoint = torch.load(opt.load_model_path)
    coop_model.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'])
    del checkpoint
    if (not opt.is_distributed) or (dist.get_rank() == 0):
        logging.info('load over from: {}'.format(opt.load_model_path))

coop_model.to(device)
# NOTE: only give prompt_learner to the optimizer
optimizer = torch.optim.Adam(
            coop_model.prompt_learner.parameters(),
            lr=opt.learning_rate,
            eps = 1e-3,
            # weight_decay=0.2,
            # betas=(0.9, 0.98),
        )

lr_scheduler = _lr_scheduler(opt, optimizer)

# self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

# Note that multi-gpu training could be slow because CLIP's size is
# big, which slows down the copy operation in DataParallel
# device_count = torch.cuda.device_count()
# if device_count > 1:
#     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#     self.model = nn.DataParallel(self.model)

## 5. load pretrained model ##
# if opt.load_pretrained:
#     checkpoint = torch.load(opt.load_model_path)
#     model_text.load_state_dict(checkpoint['model_text_state_dict'])
#     model_image.load_state_dict(checkpoint['model_image_state_dict'])
#     del checkpoint
#     if (not opt.is_distributed) or (dist.get_rank() == 0):
#         logging.info('load over from: {}'.format(opt.load_model_path))

lowest_loss = opt.now_best

## 6. training ##
crossentropyloss = nn.CrossEntropyLoss()
epoch_iters = len(train_loader)
#ctx_dim = clip_model.ln_final.weight.shape[0]
#logit_scale = clip_model.logit_scale.exp()
#Contrast_loss = contrast_loss(opt, logit_scale, ctx_dim)

for epoch in range(opt.start_epoch, opt.epochs):
    # model_image.train()
    # model_text.train()
    coop_model.train()
    epoch_loss = 0
    period_loss = 0
    period_cnt = 0
    period_time = time.time()
    start_time = time.time()
    # for id, (img_input, nouns_input, nouns_numbers) in enumerate(train_loader):
    for id, (image_feature, label) in enumerate(train_loader):    
        optimizer.zero_grad()

        image_feature = image_feature.float()# shape:[bathsize, 3, 224, 224]
        # print(image_feature.dtype)
        label = label.to(device) # shape:[batchsize,]
        # img_input = img_input.to(device)
        # nouns_input = nouns_input.to(device)
        # nouns_numbers = nouns_numbers.to(device)

        # img_input = img_input.view((-1, opt.num_segments, 3) + img_input.size()[-2:])#[32, 8, 3, 224, 224]
        # b, t, c, h, w = img_input.size()
        # img_input = img_input.view(-1, c, h, w)

        # image_embedding = model_image(img_input)
        # image_embedding = image_embedding.view(b, t, -1)

        # text_embedding = model_text(nouns_input, nouns_numbers)

        # labels = torch.arange(opt.batchsize).to(device)
        #final_loss = Contrast_loss(image_embedding, text_embedding, labels)

        # if opt.n_verb == 1:
        #     logit_scale = clip_model.logit_scale.exp()
        #     logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
        #     loss_imgs = crossentropyloss(logits_per_image, labels)
        #     loss_texts = crossentropyloss(logits_per_text, labels)
        #     final_loss = (loss_imgs + loss_texts) / 2.0
        # else: # this part may not be used in our case --LG
        #     # select the hgihest verb logits as final logits
        #     ctx_dim = clip_model.ln_final.weight.shape[0]
        #     logit_scale = clip_model.logit_scale.exp()
        #     text_embedding = text_embedding.reshape(opt.batchsize ,opt.n_verb, ctx_dim)
        #     logits_per_image, logits_per_text = create_multiverbs_logits(image_embedding, text_embedding, logit_scale)
        #     loss_imgs = crossentropyloss(logits_per_image, labels)
        #     loss_texts = crossentropyloss(logits_per_text, labels)
        #     final_loss = (loss_imgs + loss_texts) / 2.0
        # with autocast():
        #     image = model_image(image)
        output = coop_model(image_feature)
        loss = F.cross_entropy(output, label)
        loss.backward()

        # final_loss.backward()
        optimizer.step()

        if opt.lr_sheduler != 'monitor':
            if id==0 or (id + 1) % 20 == 0:
                lr_scheduler.step(epoch + id / len(train_loader))

        epoch_loss += loss.item()
        period_loss += loss.item()
        period_cnt += 1

        if ((not opt.is_distributed) or (dist.get_rank() == 0)) and (id + 1) % opt.print_freq == 0:
            progress = (id + 1) / epoch_iters
            if (period_loss / period_cnt) < lowest_loss:
                lowest_loss = period_loss / period_cnt
                ## save models
                saving_models(opt.modeldir, "best.pth", epoch, coop_model, optimizer)
            logging.info('Train Epoch: %d  | state: %d%%  |  loss: %.6f |  now_best: %.6f |  lr: %.8f  |  period_time: %.6f' % (epoch, int( 100 * progress), period_loss / period_cnt, lowest_loss, optimizer.param_groups[0]['lr'], time.time() - period_time))
            period_loss = 0
            period_cnt = 0
            period_time = time.time()

            saving_models(opt.modeldir, "lastest.pth", epoch, coop_model, optimizer)
        torch.cuda.empty_cache()

    if (not opt.is_distributed) or (dist.get_rank() == 0):    
        logging.info("="*114)
        logging.info(" " * 20 + "EPOCH:{} LOSS:{}".format(epoch, epoch_loss / epoch_iters))
        logging.info("="*114)