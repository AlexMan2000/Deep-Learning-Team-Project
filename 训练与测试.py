from model import MyModel
from torch.optim import Adam,RMSprop,SGD,ASGD,Adagrad
import lib
from dataSet import get_data_loader
import torch.nn as nn
import torch
from dataSet import show_image,PAD_IDX,VOCAB_DIM
import matplotlib.pyplot as plt
import time

attention_type = ""
seq2seq = MyModel(GRU=True).to(lib.DEVICE)


optimizer = Adam(seq2seq.parameters(),lr=lib.LEARNING_RATE)

RMSprop_opt = RMSprop(seq2seq.parameters(),lr=lib.LEARNING_RATE)

criterion_metrics = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


training_loss_list = []
testing_loss_list = []


models_name=["Adam","RMSProp","SGD","ASGD","Adagrad"]
models_list = [MyModel(attention_type="local",attention_dim=lib.ATTENTION_DIM).to(lib.DEVICE) for i in range(5)]
optimizers_list = [Adam(models_list[0].parameters(),lr=lib.LEARNING_RATE),RMSprop(models_list[1].parameters(),lr=lib.LEARNING_RATE),SGD(models_list[2].parameters(),lr=lib.LEARNING_RATE),
                   ASGD(models_list[3].parameters(),lr=lib.LEARNING_RATE),Adagrad(models_list[4].parameters(),lr=lib.LEARNING_RATE)]


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


def evaluation(epoch,model,optimizer,loss_list,model_name,display_steps=10):
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
        outputs, attention_weights = model(image, targets)

        if attention_type != "none":
            target = targets[:, 1:]  # 取<SOS>之后的文本序列,target [batch_size,seq_len]
        else:
            target = targets

        # 计算一个batch上的交叉损失，用于backpropagation
        l = criterion_metrics(outputs.view(-1, VOCAB_DIM), target.reshape(-1))

        l.backward()

        optimizer.step()

        if (idx + 1) % (display_steps) == 0:
            print("Epoch: {} training loss: {:.5f}".format(epoch + 1, l.item()))
            loss_list.append(l.item())
            # 切换成评估模式(忽略dropout)
            model.eval()
            with torch.no_grad():
                data_loader = iter(get_data_loader(train=False))
                image, caption = next(data_loader)
                image_features = model.encoder(image[0:1].to(lib.DEVICE))


                # target_caption = caption[0:1]


                if attention_type != "none":
                    captions_result, attention_weights = model.decoder \
                    .generate_caption(image_features)
                else:
                    captions_result, attention_weights = model.decoder \
                    .generate_caption(image_features.unsqueeze(0))

                # outputs,_ = model(image[0:1].to(lib.DEVICE),target_caption.to(lib.DEVICE))
                #
                # if attention_type=="none":
                #     target_caption = target_caption.to(lib.DEVICE)
                # else:
                #     target_caption = target_caption[:,1:].to(lib.DEVICE)


                # testing_loss = criterion_metrics(outputs.view(-1,VOCAB_DIM),target_caption.reshape(-1))
                # print("Epoch: {} testing loss: {:.5f}".format(epoch+1,testing_loss.item()))
                # testing_loss_list.append(testing_loss.item())

                if attention_type=="none":
                    sentence = " ".join(captions_result[1:])
                else:
                    sentence = " ".join(captions_result)

                show_image(image[0], label=sentence)
                model.train()
        save_model(model,epoch,model_name)


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



    # Model Switch
    # for i in range(5):
    #     training_loss_list = []
    #     # Epoch
    #     for t in range(2):
    #         print("testing on {}, epoch {}".format(models_name[i],t+1))
    #         evaluation(t,models_list[i],optimizers_list[i],training_loss_list,"test",10)
    #     plt.plot(list(range(len(training_loss_list))),training_loss_list,label="{}".format(models_name[i]))
    #
    # plt.xlabel("Cumulative batches")
    # plt.ylabel("Training CrossEntropyLoss")
    # plt.legend()
    # plt.show()

    # plt.plot(list(range(len(training_loss_list))),training_loss_list,label="training loss")
    # plt.plot(list(range(len(testing_loss_list))),testing_loss_list,label="testing_loss")
    # plt.xlabel("Cumulative batch")
    # plt.ylabel("CrossEntropy Loss")
    # plt.legend()
    # plt.show()

    # start = time.time()
    # for i in range(15):
    #     evaluation(i,seq2seq,optimizer,training_loss_list,"FullResNetGlobalGRU",display_steps=100)
    # plt.plot(list(range(len(training_loss_list))),training_loss_list,label="training_loss")
    # plt.plot(list(range(len(testing_loss_list))),testing_loss_list,label="testing_loss")
    # plt.xlabel("Cumulative batches/50")
    # plt.ylabel("Training CrossEntropy Loss")
    # plt.legend()
    # plt.show()
    # print("Five epoches training time: {}".format(time.time()-start))

    # model testing
    pass
