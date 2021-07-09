import torch
import torch.nn as nn
from attention import Attention
import lib
from dataSet import get_data_loader,VOCAB_DIM,PAD_IDX,dataset
from encoderCNN import MyEncoderCNN

class MyDecoderRNN(nn.Module):
    """
    This class Establish an attention-based Rnn Module for caption generation
    """
    def __init__(self,vocab_dim,
                 embedding_dim,
                 encoder_dim,
                 decoder_dim,
                 n_layers = 1,
                 dropout_p=0.5,
                 attention_dim=None,
                 attention_type = "global",
                 GRU = False):
        super().__init__()

        # 定义参数
        self.attention_type = attention_type
        self.vocab_dim = vocab_dim
        self.n_layers = n_layers
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.GRU = GRU


        # 定义embed层，将输入的caption 经过embed处理之后用于decoder生成image captions
        self.embedding_layer = nn.Embedding(num_embeddings=VOCAB_DIM,
                                            embedding_dim=embedding_dim,
                                            padding_idx=PAD_IDX)

        # 定义attention层，产生attention_score 用于后续分配注意力
        if self.attention_type == "global":
            self.attn = Attention(encoder_dim,decoder_dim)

        elif self.attention_type == "local":
            self.attn = Attention(encoder_dim,decoder_dim,attention_dim=self.attention_dim,attention_type="local")

        # 定义正则化方法
        self.dropout = nn.Dropout(dropout_p)

        # 定义初始化层
        self.init_hidden = nn.Linear(encoder_dim,decoder_dim)
        self.init_cell = nn.Linear(encoder_dim,decoder_dim)

        # 定义每一个LSTM cell单元用于手动迭代
        if self.attention_type=="local":
            self.lstm_cell = nn.LSTMCell(embedding_dim+attention_dim,decoder_dim,bias=True)
        else:
            self.lstm_cell = nn.LSTMCell(embedding_dim+encoder_dim,decoder_dim,bias=True)


        if self.GRU:
            self.gru_cell = nn.GRUCell(embedding_dim+encoder_dim,decoder_dim,bias=True)


        self.fcn = nn.Linear(decoder_dim,vocab_dim)


    def forward(self,image_features,captions):
        """

        :param image_features: encoder_outputs [batch_size,seq_len,encoder_dim]
        :param captions: numericalized captions list  [batch_size,max_len]
        :return:
        """

        embedded_captions = self.embedding_layer(captions) #[batch_size,embed_dim]

        # 初始化LSTM层
        # 对所有的features取平均用于初始化hidden_state和cell_state
        image_features_init = image_features.mean(dim=1)


        hidden_state = self.init_hidden(image_features_init)
        cell = self.init_cell(image_features_init)

        # 遍历所有时间步
        seq_len = len(captions[0])-1
        batch_size = captions.size(0)
        encoder_dim = image_features.size(1)

        # 初始化一个batch_size的所有的结果
        outputs = torch.zeros(batch_size,seq_len,self.vocab_dim).to(lib.DEVICE)
        attention_weights = torch.zeros(batch_size,seq_len,encoder_dim).to(lib.DEVICE)

        if self.GRU:
            for t in range(seq_len):
                attention_weight, context = self.attn(hidden_state, image_features)

                gru_input = torch.cat([embedded_captions[:, t], context], dim=1)

                hidden_state = self.gru_cell(gru_input, hidden_state)

                output = self.fcn(self.dropout(hidden_state))

                # 预测的词向量, output [batch_size,vocab_dim] ,attention_weight [batch_size,seq_len]
                outputs[:, t] = output
                attention_weights[:, t] = attention_weight

        else:
        #对于每一个lstm cell 我们都需要输入四个数据，hidden_state,cell,上一次 attention产生的context, 以及上一次的output(embedded之后的)
            for t in range(seq_len):

                attention_weight,context = self.attn(hidden_state,image_features)
                lstm_input = torch.cat([embedded_captions[:,t],context],dim=1)
                hidden_state, cell = self.lstm_cell(lstm_input,(hidden_state,cell))

                output = self.fcn(self.dropout(hidden_state))

                #预测的词向量, output [batch_size,vocab_dim] ,attention_weight [batch_size,seq_len]
                outputs[:,t] = output
                attention_weights[:,t] = attention_weight

        return outputs,attention_weights


    def generate_caption(self,image_features,max_len=15,vocabulary=dataset.word_dictionary):

        batch_size = image_features.size(0)

        image_features_init = image_features.mean(dim=1)
        hidden_state = self.init_hidden(image_features_init)
        cell = self.init_cell(image_features_init)


        # Starting to feed words into the RNN decoder by <SOS>
        word = torch.tensor(vocabulary.tokens_to_index["<SOS>"]).view(1,-1).to(lib.DEVICE)

        #经过embed处理
        embedded = self.embedding_layer(word)

        attention_weights_list = []
        caption_outputs = []

        #达到最大句子长度限制就停止预测
        if self.GRU:
            for i in range(max_len):
                attention_weights, context = self.attn(hidden_state, image_features)

                # store the attention weights into the list
                attention_weights_list.append(attention_weights.cpu().detach().numpy())

                gru_input = torch.cat([embedded[:, 0], context], dim=1)

                hidden_state = self.gru_cell(gru_input, hidden_state)

                # Get a list with the likelihood of each word
                output = self.fcn(self.dropout(hidden_state))  # [batch_size,vocab_dim]

                predicted_word_index = output.argmax(dim=1)  # [batch_size,1]

                caption_outputs.append(predicted_word_index.item())

                # 遇到<EOS>就停止预测
                if dataset.word_dictionary.index_to_tokens[predicted_word_index.item()] == "<EOS>":
                    break

                # for the next iteration
                embedded = self.embedding_layer(predicted_word_index.unsqueeze(0))
        else:
            for i in range(max_len):
                attention_weights,context = self.attn(hidden_state,image_features)

                #store the attention weights into the list
                attention_weights_list.append(attention_weights.cpu().detach().numpy())


                lstm_input = torch.cat([embedded[:,0],context],dim=1)


                hidden_state,cell = self.lstm_cell(lstm_input,(hidden_state,cell))
                #hidden_state [batch_size,decoder_dim]
                #cell [batch_size,decoder_dim]


                # Get a list with the likelihood of each word
                output = self.fcn(self.dropout(hidden_state)) #[batch_size,vocab_dim]

                predicted_word_index = output.argmax(dim=1) #[batch_size,1]

                caption_outputs.append(predicted_word_index.item())

                # 遇到<EOS>就停止预测
                if dataset.word_dictionary.index_to_tokens[predicted_word_index.item()] == "<EOS>":
                    break

                # for the next iteration
                embedded = self.embedding_layer(predicted_word_index.unsqueeze(0))


        caption_outputs = [dataset.word_dictionary.index_to_tokens[index] for index in caption_outputs]


        return caption_outputs,attention_weights_list



if __name__ == '__main__':
    for idx,(image,target) in enumerate(get_data_loader()):
        cnn = MyEncoderCNN()
        rnn = MyDecoderRNN(VOCAB_DIM,lib.EMBED_DIM,lib.ENCODER_DIM,lib.DECODER_DIM)
        output_image = cnn(image)
        outputs,attention_weights = rnn(output_image,target)
        print("输出",outputs,attention_weights)







