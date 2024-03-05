import torch
import os

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)#[batchsize, 512]
    x2 = x2 / x2.norm(dim=-1, keepdim=True)#[batchsize, 512]

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def create_multiverbs_logits(images, texts, logit_scale):
    '''
    images:[batchsize, 512]
    texts:[batchsize, n_verbs, 512]
    '''
    b, n_verbs, dim = texts.shape
    texts = texts / texts.norm(dim=-1, keepdim=True)#[batchsize, n_verbs, 512]
    images = images / images.norm(dim=-1, keepdim=True)#[batchsize, 512]

    # cosine similarity as logits
    texts = texts.reshape(b * n_verbs, dim) #[batchsize*n_verbs, 512]
    logits_texts = logit_scale * texts @ images.t() #[batchsize*n_verbs, batchsize]
    logits_texts = logits_texts.reshape(b, n_verbs, b) #[batchsize, n_verbs, batchsize]

    logits_images = logit_scale * images @ texts.t() #[batchsize, batchsize*n_verbs]
    logits_images = logits_images.reshape(b, b, n_verbs).permute(0, 2, 1) #[batchsize, n_verbs, batchsize]

    logits_texts = logits_texts.max(dim = 1).values
    logits_images = logits_images.max(dim = 1).values

    # shape = [global_batch_size, global_batch_size]
    return logits_texts, logits_images

def saving_models(working_dir, name, epoch, coop_model, optimizer):
    save_path = os.path.join(working_dir, name)
    torch.save({
        'epoch': epoch,
        'coop_model_state_dict': coop_model.state_dict(),
        # 'model_text_state_dict': model_text.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path, _use_new_zipfile_serialization = True)  # just change to your preferred folder/filename