import torch.nn as nn
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.encoder = model.visual

    def forward(self,image):
        return self.encoder(image)