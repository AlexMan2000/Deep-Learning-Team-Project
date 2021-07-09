import torch.nn as nn
from encoderCNN import MyEncoderCNN
from decoderRNN import MyDecoderRNN
from dataSet import VOCAB_DIM
import lib


class MyModel(nn.Module):
    """
    This class encapsulates the final encoder-decoder model.
    """
    def __init__(self,cnn_type="resnet",attention_dim=None,attention_type="global",GRU=False):
        super().__init__()
        self.encoder = MyEncoderCNN(cnn_type=cnn_type)
        self.decoder = MyDecoderRNN(
            vocab_dim=VOCAB_DIM,
            embedding_dim=lib.EMBED_DIM,
            encoder_dim= lib.ENCODER_DIM,
            decoder_dim= lib.DECODER_DIM,
            attention_dim=attention_dim,
            attention_type=attention_type,
            GRU=GRU
        )


    def forward(self,imgs,captions):
        image_features = self.encoder(imgs)
        outputs,attention_weights = self.decoder(image_features,captions)
        return outputs,attention_weights


