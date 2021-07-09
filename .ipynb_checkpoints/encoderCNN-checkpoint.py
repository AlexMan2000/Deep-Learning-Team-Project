import torch.nn as nn
import torchvision.models as models
from dataSet import get_data_loader


class MyEncoderCNN(nn.Module):
    def __init__(self,cnn_type="resnet"):
        """
        Here we are going to use the pretrained resnet50,vggnet19 model, which is capsulated in
        torchvision.models. This model is trained on mindspore dataset, which can extract
        the features in a given picture.
        """
        super().__init__()

        assert cnn_type in ["resnet","googlenet","vggnet"]
        self.cnn_type = cnn_type

        # Here we want the output from the last convolutional layer(excluding the last two fully connected layer)
        # The outputs are the features of the given images
        model = None
        if cnn_type == "resnet":
            model = models.resnet50(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            modules = list(model.children())[:-2]
            self.cnn_model = nn.Sequential(*modules)

        elif cnn_type == "googlenet":
            model = models.googlenet(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            modules = list(model.children())[:-1]
            self.cnn_model = nn.Sequential(*modules)

        elif cnn_type == "vggnet":
            model = models.vgg19(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            modules = list(model.children())[:-2]
            self.cnn_model = nn.Sequential(*modules)




    def forward(self,images):
        """
        This function takes an pre-processed image as input and produce an encoded image output from CNN for RNN input.
        :param input images: image_pixel [batch_size,3,224,224]
        :return:
        """
        if self.cnn_type=="resnet":
            output_features = self.cnn_model(images) #[batch_size,2048,7,7]

            output_features = output_features.permute(0,2,3,1) #[batch_size,7,7,2048]

            # Flatten the height and width of the image
            output_features = output_features.view(output_features.size(0),-1,output_features.size(-1)) #[batch_size,49,2048]
        elif self.cnn_type=="googlenet":
            output_features = self.cnn_model(images)
            output_features = output_features.permute(0, 2, 3, 1)  # [batch_size,7,7,2048]

            # Flatten the height and width of the image
            output_features = output_features.view(output_features.size(0), -1,
                                                   output_features.size(-1))  # [batch_size,49,2048]

        elif self.cnn_type=="vggnet":
            output_features = self.cnn_model(images)
            output_features = output_features.permute(0, 2, 3, 1)  # [batch_size,7,7,2048]

            # Flatten the height and width of the image
            output_features = output_features.view(output_features.size(0), -1,
                                                   output_features.size(-1))  # [batch_size,49,2048]

        return output_features


if __name__ == '__main__':

    for idx,(input,target) in enumerate(get_data_loader()):
        cnn = MyEncoderCNN("resnet")
        print(cnn(input))
        print(cnn(input).size())
        break