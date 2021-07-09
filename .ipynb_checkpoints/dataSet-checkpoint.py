import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose,ToTensor,Normalize,Resize,RandomCrop
import lib
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from word_dictionary import MyVocab


#Doing Data Processing
class Flickr8K(Dataset):
    def __init__(self,root_path,captions_path,transform=None,train = True):
        """
        This class create a self-defined dataset, which integrates image
        preprocessing and captions preprocessing.

        :param root_path: root file location for the data, for a more compatible transplant between group members
        :param captions_path: relative file location of captions.txt, and processed for RNN NetWork
        :param transform: Transform the image data to fit better in the CNN Network
        """

        # Predefine the some basic attribute
        self.root_path = root_path
        self.pandas_dataframe = pd.read_csv(captions_path)
        self.transform_img = transform


        # Get the images and captions
        self.imgs_list = self.pandas_dataframe["image"]
        self.captions_list = self.pandas_dataframe["caption"]


        # Getting the word_dictionary created from the input captions
        self.word_dictionary = MyVocab()
        self.word_dictionary.build_vocab(list(self.captions_list))



    def __getitem__(self,index):
        """
        Getting the single entry of the data, here in our project, we are going
        to use img as input, and caption as target label. Reference from
        torchvision.datasets.MNIST()

        :param index: int.  The index of the entry.
        :return: tuple. A single entry of the data. (img,caption)
        img: tensor object ([channels,width,height])
        caption_to_index: tensor object [3,2,4,5,1,2,....,4]
        """

        # Get the name of the image. 10...12.jpg
        img_name = self.imgs_list[index]
        # Get the path of the image.
        img_path = os.path.join(self.root_path,img_name)
        # Reference from MINST class, Implement the img_loader
        img = Image.open(img_path).convert("RGB")


        #Transform the image
        if self.transform_img:
            img = self.transform_img(img)


        # Process the captions
        caption = self.captions_list[index]

        caption_to_index = [self.word_dictionary.tokens_to_index["<SOS>"],
                            *self.word_dictionary.sentence_to_index(caption),
                            self.word_dictionary.tokens_to_index["<EOS>"]]

        return (img,torch.tensor(caption_to_index))


    def __len__(self):
        """
        :return: Number of the entries of the imgs_captions dataframe
        """
        return len(self.pandas_dataframe)



class ProcessCaption:
    """
    This class tries to fix the ingrained problem in DataLoader
    Class, where it will mistakenly recognize img_pixel as container_abcs.Sequence
    data type, which will result in the recursive pruning of the
    caption index list.
    """
    def __init__(self,pad_idx,batch_first = False):
        self.pad_idx = pad_idx
        self.batch_fist = batch_first

    def __call__(self,batch):
        """

        :param batch: An entry of data
        :return: tuple: (tensor,tensor)
        img_pixel: tensor.Size([batch_size,channels,height,width])
        caption: tensor.Size([batch_size,seq_len])
        """

        # Adding an axis at the batch_size dimension
        img_pixel = [item[0].unsqueeze(0) for item in batch]

        # Concating the tensor at the batch_size dimension
        img_pixel = torch.cat(img_pixel, dim=0)


        caption = [item[1] for item in batch]
        caption = torch.nn.utils.rnn.pad_sequence(caption, batch_first=self.batch_fist,
                                                  padding_value=self.pad_idx)

        return img_pixel, caption


DATA_PATH = "C:/Users/DELL/Desktop/NYU/NYU Class Materials/NYUSpring2021Courses/Machine Learning/Final Project/数据集"

# Used to transform the original Images
pre_transform = Compose([Resize((224,224)),
                         RandomCrop(224),
                         ToTensor(),
                         Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                         ])

dataset = Flickr8K(root_path=DATA_PATH + "/Images",
                       captions_path=DATA_PATH + "/captions.txt",
                       transform=pre_transform,
                       train = True
                       )

VOCAB_DIM = len(dataset.word_dictionary)

PAD_IDX = dataset.word_dictionary.tokens_to_index["<PAD>"]


def get_data_loader(train=True):
    """
    This function generate the dataset needed for either training set
    or testing set
    :param train:
    :return: A dataloader object(iterable)
    """
    if train:
        batch_size = lib.TRAINING_BATCH_SIZE
    else:
        batch_size = lib.TESTING_BATCH_SIZE

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=ProcessCaption(pad_idx=PAD_IDX,batch_first=True))

    return data_loader


def show_image(img_pixel,label = None):
    """
    This is an auxiliary function for
    :param img_pixel: tensor object [channels,width,height]
    """

    # Reshape the image for imshow => [width,height,channels]
    img_pixel = img_pixel.numpy().transpose((1,2,0))
    plt.imshow(img_pixel)
    if label:
        plt.title(label)
    plt.pause(0.001)
    plt.show()



if __name__ == '__main__':
    img,captions = dataset[20]
    print(show_image(img))
