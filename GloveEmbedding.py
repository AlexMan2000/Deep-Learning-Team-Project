import pandas as pd
import numpy as np
from dataSet import dataset




glove = pd.read_csv(r"F:\Downloads\glove.6B.200d.txt",sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}

def create_embedding_matrix(word_index,embedding_dict,dimension):
    embedding_matrix=np.zeros((len(word_index),dimension+4))

    for word,index in word_index.items():
        if word in embedding_dict:
            if word not in ["<UNK>","<PAD>","<SOS>","<EOS>"]:
                embedding_matrix[index]=[*embedding_dict[word],0,0,0,0]
            else:
                embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix

original_embedding_len = len(glove_embedding["<unk>"])
glove_embedding["<UNK>"] = [*glove_embedding["<unk>"],1,0,0,0]
del glove_embedding["<unk>"]
glove_embedding["<PAD>"] =  [*original_embedding_len*[0],0,0,0,1]
glove_embedding["<SOS>"] =  [*original_embedding_len*[0],0,0,1,0]
glove_embedding["<EOS>"] =  [*original_embedding_len*[0],0,1,0,0]

dictionary = dataset.word_dictionary


Glove_embedding_matrix = create_embedding_matrix(dictionary.tokens_to_index,embedding_dict=glove_embedding,dimension=200)
