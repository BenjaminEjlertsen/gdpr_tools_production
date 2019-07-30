import numpy as np
from itertools import islice

def to_features(sentence, index, we_model):
    """
    Takes a sentence, and the index of the word in the sentence that are to be converted to features.
    If a we_model (word embedding model) is fed to this method, it will also return word embeddings for the relevant
    words (current, prev and next word)
    :param sentence: The sentence the token is in
    :param index: The index of the token to return features for
    :param we_model: A word embedder model
    :return: Returns features for a token, with or without word embeddings
    """
    if we_model is None:
        return {
            'word': sentence[index].lower(),
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            #'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            #'is_all_lower': sentence[index].lower() == sentence[index],
            #'prefix-1': sentence[index][0].lower(),
            #'prefix-2': sentence[index][:2].lower(),
            'prefix-3': sentence[index][:3].lower(),
            #'suffix-1': sentence[index][-1].lower(),
            #'suffix-2': sentence[index][-2:].lower(),
            'suffix-3': sentence[index][-3:].lower(),
            'prev_word': '' if index == 0 else sentence[index - 1].lower(),
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1].lower(),
            'has_hyphen': '-' in sentence[index],
            'has_apostrophe': "'" in sentence[index],
            'has_dot': '.' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }
    else:

        we_size = we_model.getModelSize()

        word = np.zeros((we_size,)) if sentence[index].lower() not in we_model.model.wv.vocab else we_model.model.wv[
            sentence[index].lower()]

        if index == 0:
            prev_word = np.zeros((we_size,))
        elif sentence[index - 1].lower() not in we_model.model.wv.vocab:
            prev_word = np.zeros((we_size,))
        else:
            prev_word = we_model.model.wv[sentence[index - 1].lower()]

        if index == len(sentence) - 1:
            next_word = np.zeros((we_size,))
        elif sentence[index + 1].lower() not in we_model.model.wv.vocab:
            next_word = np.zeros((we_size,))
        else:
            next_word = we_model.model.wv[sentence[index + 1].lower()]

        return [word, prev_word, next_word], {
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            #'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            #'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0].lower(),
            'prefix-2': sentence[index][:2].lower(),
            'prefix-3': sentence[index][:3].lower(),
            'suffix-1': sentence[index][-1].lower(),
            'suffix-2': sentence[index][-2:].lower(),
            'suffix-3': sentence[index][-3:].lower(),
            'has_hyphen': '-' in sentence[index],
            'has_apostrophe': "'" in sentence[index],
            'has_dot': '.' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }


def get_dataset(file_path, start_index=None, batch_size=100000, we_model=None, get_words=False):
    """
    Loads data from the specified file_path. Divides it into x and y for training and validation. Is made to load
    data in chunks, using start_index and batch_size. It will load sentences from start_index until it reaches the
    batch_size. If it goes above batch_size, it will discard the latest sentence, as to avoid getting half sentences.
    Note that when calling this it will return an end_index. This should be used as start_index in next call, to read
    the data in chunks properly.
    The "class" label is transformed to "not PER" label. If other NPs are to be classified later on, the class label
    can be used to identify these.
    :param file_path: Path to the formatted and labeled data
    :param start_index: Start of chunk to load. NOTE will currently crash if no start_index is provided
    :param batch_size: Size of chunk to load
    :param we_model: Word embedding model. If not none, it is used to generate extra features.
    :param get_words: return all the words. Is useful for evaluation, as it otherwise only returns unreadable features
    :return: returns x (basic features), x_we (word embeddings for x), y, whether it reached end of the data file, and
    end index
    """
    with open(file_path) as file:
        x = []
        y = []
        sentence = []
        x_we = []
        end_of_data = True
        all_words = []
        empty_lines = 0
        end_index = start_index

        if start_index == None:
            end = None
        else:
            end = start_index + batch_size

        for line in islice(file.readlines(), start_index, end):
            word = line.split()
            end_of_data = False
            if word:
                sentence.append(word[0])
                if get_words:
                    all_words.append(word[0])
                if word[1] == 'CLASS':
                    word[1] = '0'
                y.append(word[1])

            else:
                for i in range(len(sentence)):
                    w_embeddings, vectorized_features = to_features(sentence, i, we_model)
                    x_we.append(w_embeddings)
                    x.append(vectorized_features)
                empty_lines += 1

                sentence = []

        end_index += len(x) + empty_lines

        if get_words:
            return x, x_we, y[:len(x)], end_of_data, end_index, all_words[:len(x)]

        return x, x_we, y[:len(x)], end_of_data, end_index
