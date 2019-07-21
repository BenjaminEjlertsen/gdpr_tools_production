import re

def tokenizer(text):
    """
    Splits a string into tokens, and returns them in a list. The tokens are split as whole words, and some special
    characters get their own token. For example:
    In: "The quick brown fox, jumps over the lazy dog."
    Out: "[The] [quick] [brown] [fox] [,] [jumps] [over] [the] [lazy] [dog] [.]"
    Tokenize words in a string
    :param text: String
    :return: List of tokens
    """
    return re.findall(r'(\w+-?\w*|\.|,|\(|\)|"|!|\?|\'|:|\n|/|&|-)', text)