from gensim.models import Word2Vec
from operator import itemgetter
import numpy as np
import sys

class WordEmbedder():

    def __init__(self, sentences=None, minCount = 5, algorithm = 0, windowSize = 5, neurons = 150, workers = 3):
        """
        A word embedder that uses the Word2Vec model (2 layer NN). This model takes in a one-hot word vector
        (a target word), and returns the probability of each word in the vocabulary being within the window of the this
        word. (How likely another word is used in the target words context). Each word will be embedded by the hidden
        layer. This embedding is used to define the "meaning" of a word, and similar words should be clustered.
        :param minCount: Minimum times a token(word) has to appear, to be part of the final vocabulary
        :param algorithm: Whether to use CBOW (0) or skip gram (1)
        :param windowSize: Number of context words for each side.
        :param neurons: Size of the hidden layer, and therefor final word vector
        :param workers: Number of parallel threads running the algorithm
        """
        self.model = Word2Vec(sentences=sentences, min_count=minCount, sg=algorithm, window=windowSize, size=neurons, workers=workers)

        self.femaleNameVector = None
        self.maleNameVector = None
        self.lastNameVector = None

    def train(self,sentences, totalExamples = None, totalWords = None, epochs = None):
        #Sentences should be tokenized, and contain no whitespaces. Eg. "New York" should be "New" and "York" or "New_York"
        #Consider stemming words
        #The model is case_insensitive
        self.model.train(sentences, total_examples=totalExamples, total_words=totalWords, epochs=epochs)

    def save(self,fname):
        """
        Save model
        :param fname: Location to save to
        """
        self.model.save(fname)

    def loadModel(self,fname):
        """
        Load pre-trained model

        :param fname: Location of saved model
        """
        self.model = Word2Vec.load(fname)

    def getNSimilarWords(self,word,topN = 5):
        """
        Get n most similar words to input word
        :param word: Input word
        :param topN: Number of similar words
        :return: Returns list of n similar words with respect to input word
        """
        return self.model.wv.most_similar(word,topn=topN)

    def buildVocabulary(self,sentences, updateVocab = False):
        """
        Build the vocabulary of the model (to get features). This is good when memory is an issue, and building the
        vocab over multiple iterations is the way to go.
        Note that, if no vocab has been built, update must be false. If a vocab is already built, and you want to
        expand/update it, update must be true
        :param sentences: Sentences of which the vocabulary is built on
        :param updateVocab: Update a already built vocabulary
        """
        self.model.build_vocab(sentences, update=updateVocab)

    def getVocabulary(self):
        """
        Get Models known vocabulary
        :return: List of all words in vocabulary
        """
        return list(self.model.wv.vocab)

    def getVocabularySize(self):
        """
        Get number of words in vocabulary
        :return: Int number of words in vocabulary
        """
        return len(self.getVocabulary())

    def getCorpusCount(self):
        """
        Get the number of sentences the model expects that it is built for. Note that building the vocab over multiple
        iterations, this will be equal to the number of sentences in the latest vocab build, and NOT the total
        :return: Int number of sentences
        """
        return self.model.corpus_count

    def getModelIter(self):
        """
        Need to look into this function
        """
        return self.model.iter

    def getModelSize(self):
        """
        Get the size of the hidden layer (word vector) in the model
        :return: Int hidden layer size
        """
        return self.model.layer1_size

    def getVocabularyByFrequency(self):
        """
        Get words in vocabulary, sorted by the frequency of times they appear in input text (for training)
        ### Possibly expand by also returning the actual frequency ###
        :return: Get list of most frequent words from training data
        """
        #TODO Easier way to calc:
        """mostFrequentWords = []
        for i in range(2000,2010):
            mostFrequentWords.append(model.model.wv.index2word[i])
        """
        wcounts = []
        wwordcounts = []
        for word, vocab_obj in self.model.wv.vocab.items():
            wcounts.append(vocab_obj.count)
            wwordcounts.append(word)
        meta_lst = list(enumerate(wcounts))
        sorted_meta_lst = sorted(meta_lst, key=itemgetter(1))
        sorted_meta_lst = list(reversed(sorted_meta_lst))
        sortedWords = []
        for i in range(len(sorted_meta_lst)):
            sortedWords.append(wwordcounts[sorted_meta_lst[i][0]])

        return sortedWords

    def predictNames(self, tokens, firstNameThreshold = 0.8, lastNameThreshold = 0.5, commonFname ="sofie",
                     commonMname = "peter", commonLname = "nielsen", verbose = 0):
        """
        Finds names in tokens, by finding 100 similar names to most common names. The method then sums all the vectors
        for each type (female, male, lastname) of name. The vector is scaled down, and then the dot product is
        computed between the name vectors and the input token. If the output is greater than the threshold,
        the token is added to the list of predicted names. The method furthermore keeps track of unseen tokens,
        as it cannot input unseen tokens into the model.

        :param tokens: Input text
        :param firstNameThreshold: How similar a token should be to the common name, to be considered a name
        :param lastNameThreshold: How similar a token should be to the common name, to be considered a name
        :param commonFname: A common female name, to build general vector
        :param commonMname: A common male name, to build general vector
        :param commonLname: A common lastname, to build general vector
        :param verbose: Get progress feedback
        :return: List of predicted names
        """
        if(verbose):
            print("Calculating name vectors for f = {}, m = {} and l = {}...".format(commonFname,
                commonMname, commonLname),end="")

        femaleNameVector, maleNameVector, lastNameVector = self.getNameVectors(commonFname ="sofie",
                     commonMname = "peter", commonLname = "nielsen")

        w2vNames = []
        unseen = 0
        if(verbose):
            print("DONE")
            print("Predicting names in tokens:")
            sys.stdout.flush()
        for i, token in enumerate(tokens):
            if (token != "\n"):
                try:
                    if (femaleNameVector.dot(self.model[token.lower()]) > firstNameThreshold):
                        # print("{} is a female name!".format(token))
                        w2vNames.append(i)

                    elif (maleNameVector.dot(self.model[token.lower()]) > firstNameThreshold):
                        # print("{} is a male name!".format(token))
                        w2vNames.append(i)

                    elif (lastNameVector.dot(self.model[token.lower()]) > lastNameThreshold):
                        # print("{} is a lastname!".format(token))
                        w2vNames.append(i)
                except:
                    # print("{} skipped, as it is not in the vocabulary.".format(token))
                    unseen += 1

            if(verbose and i % 100000 == 0):
                sys.stdout.flush()
                print('{}\r'.format(i/len(tokens)))


        if(verbose):
            print("{} of {} tokens not in vocabulary".format(unseen, len(tokens)))
            print("DONE")

        return w2vNames

    def getNameVectors(self, commonFname ="sofie", commonMname = "peter", commonLname = "nielsen"):
        """
        Return name vectors. Finds names in tokens, by finding 100 similar names to most common names.
        The method then sums all the vectors for each type (female, male, lastname) of name.
        The vector is then scaled down.
        :param commonFname: A common female name, to build general vector
        :param commonMname: A common male name, to build general vector
        :param commonLname: A common lastname, to build general vector
        :return: A name vector for each type (female, male, lastname) of name
        """
        femaleNames = self.getNSimilarWords(commonFname, topN=100)
        femaleNameVector = np.zeros(self.getModelSize(), )
        for i in range(len(femaleNames)):
            femaleNameVector += self.model[femaleNames[i][0]]

        femaleNameVector /= 1000

        maleNames = self.getNSimilarWords(commonMname, topN=100)
        maleNameVector = np.zeros(self.getModelSize(), )
        for i in range(len(maleNames)):
            maleNameVector += self.model[maleNames[i][0]]

        maleNameVector /= 1000

        lastnames = self.getNSimilarWords(commonLname, topN=100)

        lastNameVector = np.zeros(self.getModelSize(), )
        for i in range(len(lastnames)):
            lastNameVector += self.model[lastnames[i][0]]

        lastNameVector /= 1000

        return femaleNameVector, maleNameVector, lastNameVector