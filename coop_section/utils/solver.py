import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

def _optimizer(opt, model_image, model_text):
    clip_model_parameters = []
    anti_clip_model_parameters = []
    # for name, parameter in fusion_model.named_parameters():
    #     anti_clip_model_parameters.append(parameter)
    for name, parameter in model_text.named_parameters():
        if name != 'module.prompt_learner.ctx':
            clip_model_parameters.append(parameter)
        else:
            anti_clip_model_parameters.append(parameter)
    for name, parameter in model_image.named_parameters():
        clip_model_parameters.append(parameter)

    if opt.optim == 'adam':
        optimizer = optim.Adam([
            {'params': clip_model_parameters, 'lr': opt.clip_learning_rate}, 
            {'params': anti_clip_model_parameters, 'lr': opt.learning_rate}],
            lr = opt.learning_rate, betas = (0.9, 0.98), eps = 1e-8, weight_decay = 0.2
        )
        # optimizer = optim.Adam([
        #     {'params': model.parameters()},  
        #     {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
        #                        lr=config.solver.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    # elif config.solver.optim == 'sgd':

    #     optimizer = optim.SGD([{'params': model.parameters()},  
    #      {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
    #                           config.solver.lr,
    #                           momentum=config.solver.momentum,
    #                           weight_decay=config.solver.weight_decay)
    #     print('SGD')
    # elif config.solver.optim == 'adamw':
    #     vision_params = list(map(id, model.visual.parameters()))
    #     text_params = filter(lambda p: id(p) not in vision_params,
    #                          model.parameters())

    #     optimizer = optim.AdamW([{'params': text_params},
    #                              {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
    #                              {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
    #                             betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
    #                             weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    #     for param_group in optimizer.param_groups:
    #         print(param_group['lr'])
    #     print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(opt.optim))
    return optimizer

def _lr_scheduler(opt, optimizer):
    if opt.lr_sheduler == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            opt.lr_sheduler_per_epoch,
            warmup_epochs=opt.lr_warmup_step
        )
    # elif config.solver.type == 'multistep':
    #     if isinstance(config.solver.lr_decay_step, list):
    #         milestones = config.solver.lr_decay_step
    #     elif isinstance(config.solver.lr_decay_step, int):
    #         milestones = [
    #             config.solver.lr_decay_step * (i + 1)
    #             for i in range(config.solver.epochs //
    #                            config.solver.lr_decay_step)]
    #     else:
    #         raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
    #     lr_scheduler = WarmupMultiStepLR(
    #         optimizer,
    #         milestones,
    #         warmup_epochs=config.solver.lr_warmup_step
    #     )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(opt.lr_sheduler))
    return lr_scheduler