import spacy


#Using the fastest word splitting tools
SPACY_OBJ = spacy.load("en_core_web_sm")

#Defining our dictionary for all the words contained in the provided captions
class MyVocab:
    """
        This class is responsible for constructing the dictionary which contains
        all the words that appear over a certain frequency, which we will use to
        tokenize any given sentence for our RNN model.

    """
    def __init__(self):
        #Pre restore the tokens mapping.
        """
        These are severals pre-defined tokens to pre process the sentence(sequqnce).
        <PAD>: Used to pad any given sentence to a uniform length, making it easier for
        RNN model to handle.
        <SOS>: Inserted at the start of each sentence.
        <EOS>: Appended at the end of each sentence.
        <UNK>: Mark the word that hasn't appeared in the captions in the training data.
        """
        self.index_to_tokens = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}

        #Inverse the above dictionary
        self.tokens_to_index = {value:key for key,value in self.index_to_tokens.items()}



    def __len__(self):
        """
        :return: int The number of the stored tokens
        """
        return len(self.index_to_tokens)


    def build_vocab(self,sentence_list,min_count=1,max_count=None,max_features=None):
        """
        This function builds the dictionary for RNN model
        :param sentence_list: An iterable containers that includes all the sentences
        :param min_count: The minimum number of the time that a word should appear in all the sentences.
        :param max_count: The maximum number of the time that a word should appear in all the sentences.
        :param max_features: Number of words to keep(From the most frequent words).
        :return:
        """

        #Create a dictionary for counting word frequency
        self.frequency_counter = {}

        #Create word_dict from several sentences
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                self.frequency_counter[word] = self.frequency_counter.get(word,0)+1

        #Filtering
        if min_count is not None:
            self.frequency_counter = {word:value for word,value in self.frequency_counter.items() if value >= min_count}

        if max_count is not None:
            self.frequency_counter = {word:value for word,value in self.frequency_counter.items() if value <= max_count}

        if max_features is not None:
            self.frequency_counter = dict(list(sorted(self.frequency_counter.items(),key=lambda x:x[-1],reverse=True))[:max_features])

        #Creating words_to_index mapping
        for word in self.frequency_counter:
            self.tokens_to_index[word] = len(self.tokens_to_index)


        #Creating index_to_words mapping
        self.index_to_tokens = dict(zip(self.tokens_to_index.values(),self.tokens_to_index.keys()))



    def sentence_to_index(self,sentence,max_len = 20):
        """
        This function converts the sentence to word index and controls
        the maximum length of the sentence. Meanwhile, it adds <SOS> and <EOS> tags
        to the beginning and the ending of a given sentence.
        :param sentence: string, A sentence in string.
        :param max_len: int, performing sentence pruning
        :return:
        """
        tokenized_sentence = self.tokenize(sentence)
        if max_len is not None:
            tokenized_sentence = tokenized_sentence[:max_len]

        return [self.tokens_to_index.get(word,self.tokens_to_index["<UNK>"]) for word in tokenized_sentence]


    def index_to_sentence(self,indices):
        """
        :param indices: A list of the index of words, e.g. [2,3,6,9,10,...]
        :return: A list of word corresponding to the indices, ["today","good","date",...]
        """
        return [self.index_to_tokens.get(index) for index in indices]



    @staticmethod
    def tokenize(content):
        # Filter out the unwanted signs(Unlikely to be seen in a generated sentence.)
        tokens = [token.text.lower() for token in SPACY_OBJ.tokenizer(content)]
        return tokens


if __name__ == '__main__':

    v = MyVocab()

    v.build_vocab(["A group of people is playing on the playground"])

    print(v.tokens_to_index)
    print(v.sentence_to_index("What is wrong with you!!!"))