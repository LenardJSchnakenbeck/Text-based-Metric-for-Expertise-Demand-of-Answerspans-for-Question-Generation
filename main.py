import pandas as pd
import spacy

articles = pd.read_json("preprocessing/wikipedia_texts.json")#, encoding="utf-8")


def get_wikipedia_text(i, articles = articles):
    return articles.iloc[0]["text"]

"""
def noun_phrase_chunking(text):
    nlp = spacy.load("en_core_web_sm")
    a = []
    chunked_text = nlp(text)
    noun_chunks = []
    for chunk in chunked_text.noun_chunks:
        noun_chunks += [chunk.text]
    tokenized_text = nltk.word_tokenize(str(text))
    for i in len(tokenized_text):
        if tokenized_text[i] in noun_chunks[0]:
            noun_chunk = i

        else:
            noun_chunk = None

        a += [tokenized_text[i],noun_chunk]

    return
"""

a = []
for i in range(len(articles)):
    a += [get_wikipedia_text(i, articles=articles)]





#main
# TODO: FILESYSTEM

# TODO: 2 Sätze schreiben
# TODO: noun phrase labelling
# TODO: tokenization (1word chunking)

#Relevance
# TODO: SingleRank
# TODO: FAST  KeyBERT
# TODO: SAKE

#Similarity of false Alternatives
# TODO: Entity Coreference labelling
# TODO: Similarity via BERT-Embeddings considering Entity Coreferences
