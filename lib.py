import torch

# Data Information


TRAINING_BATCH_SIZE = 156

TESTING_BATCH_SIZE = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_DIM = 2048 #由图片的最后一个维度决定

DECODER_DIM = 512

ATTENTION_DIM = 200

LEARNING_RATE = 0.001

EMBED_DIM = 200

