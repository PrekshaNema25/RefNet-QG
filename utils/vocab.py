import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer 
from gensim.models import Word2Vec
import gensim
from collections import defaultdict
from math import sqrt
import numpy as np 
import sys
import codecs
import operator

class Vocab():
    def __init__(self, len_vocab=0):

        """ Initalize the class parameters to default values
        """

        self.pos_to_index = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.index_to_pos = {}
        self.char_to_index = {}
        self.unknown       = "<unk>"
        self.unknown_char  = "$"
        self.end_of_sym    = "<eos>"
        self.start_sym     = "<s>"
        self.padding       = "<pad>"
        self.avg_freq      = 0 
        self.word_freq     = {}
        self.len_vocab     = len_vocab
        self.total_words   = 0
        self.embeddings    = None


    def get_global_embeddings(self, filenames, embedding_size, embedding_dir):

        """ Construct the Embedding Matrix for the sentences in filenames.

            Args:
                filenames: File names of the training files.
                embedding_size: Dimensions for the embedding to be used.

            Returns
                Embedding matrix.
        """
        sentences = []
        print (embedding_dir)

        if (os.path.exists(os.path.join(embedding_dir , 'vocab_len.pkl'))):
            vocab_len_stored = pickle.load(open(os.path.join(embedding_dir , "vocab_len.pkl"), "rb"))
        else:
            vocab_len_stored = 0

        if (os.path.exists(os.path.join(embedding_dir , "embeddings_{}.pkl".format(self.len_vocab)))):
            print ("Load file")
            self.embeddings = pickle.load(open(os.path.join(embedding_dir , "embeddings_{}.pkl".format(self.len_vocab)), "rb"))
            return None

        print (os.path.exists(os.path.join(embedding_dir, 'embeddings')))
        if (os.path.exists(os.path.join(embedding_dir , 'embeddings')) == True):
            #model = gensim.models.KeyedVectors.load_word2vec_format('../Data/embeddings.bin', binary = True)
            #model = Word2Vec.load_word2vec_format('../Data/embeddings.bin', binary = True)
            print ("Yay")
            model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(embedding_dir , 'embeddings'), binary=False)
            print ("DEBUG: Pretrained Embedding Loaded")
        else:
            for file in filenames:
                with open(file, 'rb') as f:
                    for lines in f:
                        words = [lines.split()]
                        sentences.extend(words)

            model = Word2Vec(sentences, size=embedding_size, min_count=0)
            model.save(embedding_dir + "embeddings.bin")

        self.embeddings_model = model
        return model


    def add_constant_tokens(self):

        """ Adds the constant tokens in the dictionary.
        """

        self.word_to_index[self.padding]    = 0
        self.word_to_index[self.unknown]    = 1
        self.word_to_index[self.start_sym]   = 2
        self.word_to_index[self.end_of_sym] = 3

        self.pos_to_index[self.padding]    = 0
        self.pos_to_index[self.unknown]    = 1
        self.pos_to_index[self.start_sym]   = 2
        self.pos_to_index[self.end_of_sym] = 3

        self.char_to_index[' '] = 0

    def add_word(self, word):

        """ Adds the word to the dictionary if not already present.

        Arguments:
            * word : Word to be added.

        Returns:
            * void
        """
        if word in self.word_to_index:
            self.word_freq[word] = self.word_freq[word] + 1

        elif word == "<pad>":
            return
        else:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.word_freq[word] = 1
    

    def add_char(self, char):
        if char in self.char_to_index: 
            pass
        else:
            index = len(self.char_to_index)
            self.char_to_index[char] = index

        return

    def create_reverse_dictionary(self):

        """ Creates a mapping from index to the words
            This will be used during the time of decoding the
            indices to words.
        """
        for key in self.word_to_index:
            self.index_to_word[self.word_to_index[key]] = key

        for key in self.pos_to_index:
            self.index_to_pos[self.pos_to_index[key]] = key

    def construct_dictionary_single_file(self, filename):
        
        """ Creates the dictionary from a single file.

            Arguments:
                * filename: The file from which dictionary
                          will be constructed

            Returns:
                * void
        """
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            for lines in f:
                for words in lines.lower().split():
                    self.add_word(words.lower())
                    for char in words:
                        self.add_char(char.lower())

    def construct_pos_dictionary_file(self, filename):
   	print (filename) 
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            for lines in f:
                for pos_tag in lines.strip().split():
                
                    if pos_tag not in self.pos_to_index:
                        index = len(self.pos_to_index)
                        self.pos_to_index[pos_tag] = index


    def fix_the_frequency(self, limit=0):

        temp_word_to_index = {}
        temp_index_to_word = {}

        #get the list of the frequent words, upto the given limit.
        word_list = []
        count = 0

        sorted_word_to_index = sorted(self.word_freq.items(), key = operator.itemgetter(1), reverse=True)
        new_index = 4 

        for k, v in sorted_word_to_index:
            temp_word_to_index[k] = new_index
            temp_index_to_word[new_index] = k
            new_index += 1
            if (limit <= len(temp_word_to_index)):
                break
        print (len(temp_word_to_index))
        self.word_to_index = temp_word_to_index


    def construct_dictionary_multiple_files(self, filenames):

        """ Dictionary is made from all the files listed.

            Arguments :
                * filenames = List of the filenames 

            Returns :
                * None
        """

        for files in filenames:
            self.construct_dictionary_single_file(files)

    def encode_word(self, word):

        """ Conver the word to the particular index

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """
        if word not in self.word_to_index:
            word = self.unknown

        return self.word_to_index[word]

    def encode_pos(self, pos):

        """ Conver the word to the particular index

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """
        if pos not in self.pos_to_index:
            pos = self.unknown

        return self.pos_to_index[pos]


    def encode_char(self, char):
        if char not in self.char_to_index:
            char = self.unknown_char
        return self.char_to_index[char]


    def decode_word(self, index):

        """ Index is converted to its corresponding word.

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_word:
            return self.unknown
        return self.index_to_word[index]


    def decode_pos(self, index):

        """ Index is converted to its corresponding word.

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_pos:
            return self.unknown
        return self.index_to_pos[index]


    def get_embeddings_pretrained(self, embedding_size, embedding_dir):

        """ Embeddings are appened based on the index of the 
        word in the matrix self.embeddings.
        """

        sorted_list = sorted(self.index_to_word.items(), key = operator.itemgetter(0))
        np.random.seed(1357)


        if (os.path.exists(embedding_dir + 'vocab_len.pkl')):
            vocab_len_stored = pickle.load(open(embedding_dir + "vocab_len.pkl", "rb"))
        else:
            vocab_len_stored = 0
        
        if os.path.exists(embedding_dir + "embeddings_{}.pkl".format(self.len_vocab)):
            self.embeddings = pickle.load(open(embedding_dir + "embeddings_{}.pkl".format(self.len_vocab), "rb"))
            return

        embeddings = []
        count = 0
        for index, word in sorted_list:

            try:
                self.embeddings_model

                if word in self.embeddings_model.vocab:
                    count = count + 1 
                    embeddings.append(self.embeddings_model[word])
                else:
                    if word in ['<pad>', '<s>', '<eos>']:
                            temp = np.zeros((embedding_size))

                    else:
                            temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size), (embedding_size))

                    embeddings.append(temp)

            except:
                if word in ['<pad>', '<s>', '<eos>']:
                    temp = np.zeros((embedding_size))
                else:
                    temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size), (embedding_size))

                embeddings.append(temp)

        print ("Number of words in the count" , count)
        self.embeddings = np.asarray(embeddings)
        self.embeddings = self.embeddings.astype(np.float32)


        pickle.dump(self.embeddings, open(embedding_dir + "embeddings_{}.pkl".format(self.len_vocab), "wb"))
        pickle.dump(self.len_vocab, open(embedding_dir + "vocab_len.pkl", "wb"))


    def construct_vocab(self, args, filenames, pos_dir,embedding_size=100, vocab_length=74, embedding_dir="../Data/"):


        """ Constructs the embeddings and  vocabs from the parameters given.

            Args:
                * filenames: List of filenames to consider to make the vocab
                * embeddings: The size of the embeddings to be considered.

            Returns:
                * void
        """

        self.construct_dictionary_multiple_files(filenames)
        self.fix_the_frequency(vocab_length)
        print (filenames, vocab_length)
        print ("Length of the dictionary is " + str(len(self.word_to_index)))
        sys.stdout.flush()
        self.add_constant_tokens()
        if args["Hyperparams"]["use_pos_decoder"] == "True":
            self.construct_pos_dictionary_file(pos_dir)
	    print ("<=>"*30,len(self.pos_to_index))
            for (pos,idx) in self.pos_to_index.items():
                print ("Index: ",idx,"POS: ",pos)
        self.create_reverse_dictionary()
        self.get_global_embeddings(filenames, embedding_size, embedding_dir)
        self.get_embeddings_pretrained(embedding_size, embedding_dir)

        print ("DEBUG: Vocab Embedding Shape is {} ".format(self.embeddings.shape))
        self.len_vocab = len(self.word_to_index)
        self.len_vocab_char = len(self.char_to_index)
        self.avg_freq = np.average([self.word_freq[i] for i in self.word_freq])
        print ("DEBUG: Length of the dictionaries  is Word {} and Char {}".format(len(self.word_to_index), len(self.char_to_index)))
        print ("DEBUG: Some Samples")
        count = 0 
        for k in self.word_to_index:
            #print (k)
            if count == 10:
                break
            count += 1

        #print ("Character Vocab" , self.char_to_index)
        self.total_words = float(sum(self.word_freq.values()))


    def plot_the_frequencies(self):
        
        x = self.index_to_word.keys()
        y = []

        for i in self.index_to_word.values():
            if i not in self.word_freq:
                y.append(0)
            else:
                y.append(self.word_freq[i])


        plt.hist(x, weights=y)
        plt.show()

def main():
    vocab_length = 30000
    x = Vocab(vocab_length)
    filenames = ["../Gigaword/train_title", "../Gigaword/train_content"]
    args = {}
    args["Hyperparams"] = {}
    args["Hyperparams"]["use_pos_decoder"] = "False" 
    working_dir = "squad_1.1/sentence_level/"
    filenames = [working_dir + "train_summary" , working_dir + "train_content", working_dir + "train_query"]
    pos_dir = working_dir + "train_pos_summary"
    embedding_dir = "embeddings_sent"
    embedding_size = 300
    x.construct_vocab(args,filenames,pos_dir,embedding_size,vocab_length,embedding_dir)
    word_to_index = x.word_to_index
    sorted_word_to_index = sorted(word_to_index.items(), key=operator.itemgetter(1)) 
    vocab_words = [ i[0] for i in sorted_word_to_index ] 
    pickle.dump(vocab_words,open("vocab_words/squad_1.1_sent.pkl","wb"))    
    #print (sorted_word_to_index)
    #x.plot_the_frequencies()

    #for i in x.word_to_index:
    #    print (i, x.word_freq[i])


if __name__ == '__main__':
    main()
