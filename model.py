import torch.nn as nn
from encoderCNN import MyEncoderCNN
from decoderRNN import MyDecoderRNN
from dataSet import VOCAB_DIM
import lib


class MyModel(nn.Module):
    """
    This class encapsulates the final encoder-decoder model.
    """
    def __init__(self,cnn_type="resnet",attention_dim=None,attention_type="global",embedding_type="randomize",embedding_dim=lib.EMBED_DIM,GRU=False,encoder_dim=lib.ENCODER_DIM,decoder_dim = lib.DECODER_DIM):
        super().__init__()

        self.attention_type = attention_type
        self.encoder = MyEncoderCNN(cnn_type=cnn_type,attention_type=attention_type,embed_dim=embedding_dim)
        self.decoder = MyDecoderRNN(
            vocab_dim=VOCAB_DIM,
            embedding_dim=lib.EMBED_DIM,
            encoder_dim= encoder_dim,
            decoder_dim= decoder_dim,
            attention_dim=attention_dim,
            attention_type=attention_type,
            embedding_type="randomized",
            GRU=GRU
        )


    def forward(self,imgs,captions):
        image_features = self.encoder(imgs)
        if self.attention_type=="none":
            outputs,_ = self.decoder(image_features,captions)
            return (outputs,None)
        else:
            outputs,attention_weights = self.decoder(image_features,captions)
            return outputs,attention_weights


