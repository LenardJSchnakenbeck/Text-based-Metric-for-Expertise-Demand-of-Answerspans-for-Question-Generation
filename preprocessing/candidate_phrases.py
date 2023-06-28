import nltk
import pandas as pd
import json




def get_noun_phrases(document):
    return document["nounphrases_mock"]


def get_tokens(document):
    return nltk.word_tokenize(document['text'])


def append_tokens(documents):
    labeled_documents = []
    for document in documents:
        noun_phrases = get_noun_phrases(document)
        tokens = get_tokens(document)
        document['nounphrases'] = noun_phrases
        document['tokens'] = tokens
        labeled_documents.append(document)
    return labeled_documents

def run():
    text_labeled = open('wikipedia_texts.json', encoding="utf-8")
    documents = json.load(text_labeled)
    with open('labeled_wikipedia_texts.json', 'w') as outfile:
        json.dump(append_tokens(documents), outfile)

run()