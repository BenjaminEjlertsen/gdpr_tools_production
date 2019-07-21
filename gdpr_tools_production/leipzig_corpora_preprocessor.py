from gdpr_tools_production.nlp_tools import tokenizer
import sys

def get_preprocessed_corpora(inputFiles, maxSentences=None, tokenize = True):
    """
    Load the text from the Leipzig Corpora. All tabs (\t) and newlines (\s) are removed, as the corpora are already
    split into sentences. Furthermore, the indices of the sentences are removed.
    :param inputFiles: A list of the file locations for extracting data (formatted like the Leipzig Corpora)
    :param maxSentences: Get n sentences, rather than the current 5.3 million sentences
    :param tokenize: Whether the output should be tokenized
    :return: List of all sentences, optionally tokenized.
    """
    preprocessedSentences = []

    #Did this, as we might automatically convert a single file into a list later? Otherwise, delete
    listOfInputFiles = [inputFiles]

    print("Cleaning data...",end="")

    print("Preprocessing corpora: ", listOfInputFiles)
    sys.stdout.flush()
    for cfile in listOfInputFiles:
        try:
            with open(cfile) as corpusFile:
                tmpIndex = ""

                corpus = corpusFile.readlines()
                for line in corpus:
                    if(maxSentences != None):
                        if(maxSentences <= 0): break
                    foundIndex = False
                    cleanedSentence = ""
                    tmpIndex = ""
                    for char in line:
                        if not foundIndex and char.isdigit():
                            tmpIndex += char
                        else:
                            foundIndex = True
                            cleanedSentence += char

                    cleanedSentence = cleanedSentence.replace("\t", "")
                    preprocessedSentences.append(cleanedSentence.replace("\n",""))
                    if(maxSentences != None):
                        maxSentences -= 1
        except:
            print("Could not open file")

    if(tokenize):
        for i, line in enumerate(preprocessedSentences):
            preprocessedSentences[i] = tokenizer(preprocessedSentences[i].lower())

    print("DONE\n")
    return preprocessedSentences