import torch
from dataSet import show_image
from lib import DEVICE
from dataSet import dataset,pre_transform
import matplotlib.pyplot as plt
from dataSet import get_data_loader
from model import MyModel
from PIL import Image
import pandas as pd
import lib


attention_type=""
img_caption_mapping = pd.read_csv("./数据集/grouped_img_with_captions.csv")


def get_trained_model(model_name=None):
    if model_name=="res":
        print("res")
        model = MyModel(GRU=True,encoder_dim=2048).to(DEVICE)
        tmp = torch.load("./模型存放/FullResNetGlobalGRU.pkl")
        model.load_state_dict(tmp["model_state_dict"])
    else:
        print("vgg")
        model = MyModel(cnn_type="vggnet",attention_type="local",attention_dim=128,encoder_dim=512,decoder_dim=400).to(DEVICE)
        tmp = torch.load("./模型存放/FullVGGLocalLSTM.pkl")
        model.load_state_dict(tmp["model_state_dict"])
    return model



def get_trained_model_simple():
    model = MyModel(GRU=True,cnn_type="vggnet",encoder_dim=512,attention_type="local",decoder_dim=256,attention_dim=128).to(DEVICE)
    tmp = torch.load("./模型存放/NewVGGLocalLSTM.pkl")
    model.load_state_dict(tmp["model_state_dict"])
    return model



def get_caps_from(features_tensors):
    model = get_trained_model_simple()
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(DEVICE))
        caps, alphas = model.decoder.generate_caption(features, vocabulary=dataset.word_dictionary)
        caption = ' '.join(caps)

        show_image(features_tensors[0], label=caption)
    return caps, alphas


def get_caps_from_overload(img_path,features_tensors,model_name):
    model = get_trained_model(model_name)
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(DEVICE))

        if attention_type!="none":
            caps, alphas = model.decoder.generate_caption(features, vocabulary=dataset.word_dictionary)
        else:
            caps,_ = model.decoder.generate_caption(features.unsqueeze(0),vocabulary=dataset.word_dictionary)

        caption = ' '.join(caps)


        # 获取真实的caption

        img_name = img_path.split("/")[-1]
        true_captions = img_caption_mapping[img_caption_mapping.image == img_name].caption_list.values
        # print(true_captions)

        show_image(features_tensors[0], label=caption, img_name = img_name)

    if attention_type=="none":
        return caps[1:],None
    else:
        return caps, alphas


def visualize_attention(img,result,attention_plot,img_path=None):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    if attention_plot == None:
        for l in range(len_result):
            ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
            ax.set_title(result[l])
            ax.imshow(temp_image)
    else:
        print("有注意力")
        for l in range(len_result):

            temp_att = attention_plot[l].reshape(7, 7)

            ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    if img_path:
        img_name = img_path.split("/")[-1]
        plt.savefig("./static/{}{}".format("~attention_image~",img_name))
    plt.show()


def show_self_collected_image(image_path,model_name):
    img = Image.open(image_path).convert("RGB")

    #transform the image to cropped RGB
    img0 = pre_transform(img)
    img1 = pre_transform(img)
    caps,alphas = get_caps_from_overload(image_path,img0.unsqueeze(0),model_name)

    visualize_attention(img1,caps,alphas,image_path)
    caption = " ".join(caps)
    if "." in caption:
        caption = caption.split(".")[0]
    return caption


if __name__ == '__main__':
    dataiter = iter(get_data_loader(train=True))
    images, _ = next(dataiter)

    img = images[0].detach().clone()
    img1 = images[0].detach().clone()
    caps, alphas = get_caps_from(img.unsqueeze(0))
    visualize_attention(img1, caps, alphas)


    # Self-collected data
    # show_self_collected_image("./数据集/Images/390992388_d74daee638.jpg","vgg")






