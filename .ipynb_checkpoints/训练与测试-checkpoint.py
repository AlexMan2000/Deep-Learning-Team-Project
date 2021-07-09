from model import MyModel
from torch.optim import Adam,RMSprop,SGD,ASGD,Adagrad
import lib
from dataSet import get_data_loader
import torch.nn as nn
import torch
from dataSet import show_image,PAD_IDX,VOCAB_DIM
import matplotlib.pyplot as plt

seq2seq = MyModel().to(lib.DEVICE)


optimizer = Adam(seq2seq.parameters(),lr=lib.LEARNING_RATE)


criterion_metrics = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

loss_list = []

def save_model(model,epochs,model_name):
    """
    This function saves the model's parameters
    :param model:
    :param epoch:
    :return:
    """
    model_param = {
        "epochs":epochs,
        "vocab_size":VOCAB_DIM,
        "embed_size":lib.EMBED_DIM,
        "encoder_dim":lib.ENCODER_DIM,
        "decoder_dim":lib.DECODER_DIM,
        "attention_dim":lib.ATTENTION_DIM,
        "model_state_dict":model.state_dict()
    }

    torch.save(model_param,"./模型存放/{}.pkl".format(model_name))


def evaluation(epoch,model_name,display_steps=20):
    """
    This function serves to dynamically evaluate the model's performance.
    :param epoch:
    :param display_steps:
    :return:
    """

    for idx, (image, targets) in enumerate(get_data_loader(train=True)):
        # image [batch_size,seq_len,encoder_dim]
        # targets [batch_size,max_len]

        image, targets = image.to(lib.DEVICE), targets.to(lib.DEVICE)

        optimizer.zero_grad()

        # outputs [batch_size,seq_len,vocab_size]
        outputs, attention_weights = seq2seq(image, targets)

        target = targets[:, 1:]  # 取<SOS>之后的文本序列,target [batch_size,seq_len]

        # 计算一个batch上的交叉损失，用于backpropagation
        l = criterion_metrics(outputs.view(-1, VOCAB_DIM), target.reshape(-1))

        l.backward()

        optimizer.step()

        if (idx + 1) % (display_steps) == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch + 1, l.item()))
            loss_list.append(l.item())
            # 切换成评估模式(忽略dropout)
            seq2seq.eval()
            with torch.no_grad():
                data_loader = iter(get_data_loader(train=False))
                image, caption = next(data_loader)
                image_features = seq2seq.encoder(image[0:1].to(lib.DEVICE))

                captions_result, attention_weights = seq2seq.decoder \
                    .generate_caption(image_features)

                sentence = " ".join(captions_result)

                show_image(image[0], label=sentence)
            seq2seq.train()
        save_model(seq2seq,epoch,model_name)


if __name__ == '__main__':
    # for i in range(100):
    #     for idx,(image,targets) in enumerate(get_data_loader(train=True)):
    #         # image [batch_size,seq_len,encoder_dim]
    #         # targets [batch_size,max_len]
    #
    #         image,targets = image.to(lib.DEVICE), targets.to(lib.DEVICE)
    #
    #         optimizer.zero_grad()
    #
    #         # outputs [batch_size,seq_len,vocab_size]
    #         outputs, attention_weights = seq2seq(image,targets)
    #
    #         target = targets[:,1:] #取<SOS>之后的文本序列,target [batch_size,seq_len]
    #
    #         # 计算一个batch上的交叉损失，用于backpropagation
    #         l = criterion_metrics(outputs.view(-1,VOCAB_DIM),target.reshape(-1))
    #
    #         l.backward()
    #
    #         optimizer.step()
    #
    #         if (idx+1) % (10) == 0:
    #             print("Epoch: {} loss: {:.5f}".format(i+1,l.item()))
    #
    #             # 切换成评估模式(忽略dropout)
    #             seq2seq.eval()
    #             with torch.no_grad():
    #                 data_loader = iter(get_data_loader(train=False))
    #                 image, caption = next(data_loader)
    #                 image_features = seq2seq.encoder(image[0:1].to(lib.DEVICE))
    #
    #                 captions_result, attention_weights = seq2seq.decoder\
    #                                                             .generate_caption(image_features)
    #
    #                 sentence = " ".join(captions_result)
    #
    #                 show_image(image[0],label = sentence)
    #             seq2seq.train()

    for i in range(2):
        evaluation(i,"ResNetGlobalLSTM",20)

    print(loss_list)

    plt.plot(list(range(len(loss_list))),loss_list)