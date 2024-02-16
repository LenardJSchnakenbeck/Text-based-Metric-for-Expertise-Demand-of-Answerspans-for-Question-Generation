import json
import pickle
import nltk
from nltk.corpus import wordnet
import numpy as np
from itertools import chain
from preprocessing.candidate_phrases import get_main_word, map_pos_tokenizer_to_lemmatizer


def lemmatize(candidate, pos):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pos = map_pos_tokenizer_to_lemmatizer(pos)
    if pos:
        return lemmatizer.lemmatize(candidate, map_pos_tokenizer_to_lemmatizer(pos))
    else:
        return candidate

def wordnet_similarity(candidates, candidates_pos, tokens, tokens_pos):
    def check_for_others(superordinates, subordinate_function, tokens_lemmatized, candidate):
        score = 0
        # score +=1 if meronym that is no synonym and not the candidate itself
        subordinates = set(chain.from_iterable([subordinate.lemma_names()
                    for superordinate in superordinates for subordinate in eval(subordinate_function)]))
        # del synonyms
        number_synonyms = len(subordinates)
        subordinates = [subordinate.lower() for subordinate in subordinates if subordinate not in synonyms]
        number_synonyms = number_synonyms - len(subordinates)
        for token in tokens_lemmatized:
            if token.lower() in subordinates and token.lower() != candidate.lower():
                score += 1
        return score

    #adjust pos for lemmatizer
    candidates_pos = list(
        map(map_pos_tokenizer_to_lemmatizer, candidates_pos))

    tokens_pos = list(
        map(map_pos_tokenizer_to_lemmatizer, [x[-1] for x in tokens_pos]))
    tokens_lemmatized = [lemmatize(token, token_pos) for token, token_pos in zip(tokens, tokens_pos)]

    scores = []
    i = 0
    for candidate in candidates:
        if isinstance(candidate, list):
            len_can = len(candidate)
            pos = candidates_pos[i:i+len_can]

            candidate, pos = get_main_word(candidate, pos)
            synsets = wordnet.synsets(lemmatize(candidate,pos))

            synonyms = set(chain.from_iterable([word.lemma_names() for word in synsets]))
            holonyms = [holonym for synset in synsets for holonym in synset.member_holonyms() + synset.part_holonyms()]
            hypernyms = [hypernym for synset in synsets for hypernym in synset.hypernyms()]

            scores += [
                check_for_others(
                    holonyms, "superordinate.member_meronyms()", tokens_lemmatized, lemmatize(candidate,pos))
                +check_for_others(
                    hypernyms, "superordinate.hyponyms()", tokens_lemmatized, lemmatize(candidate,pos))
            ]

            print(candidate, scores)
            i += len_can
        else:
            i += 1
    return scores


def calculate_score_and_write_pickle(labeled_documents_path, wordnet_similarities_path):
    documents_json = open(labeled_documents_path)
    documents = json.load(documents_json)

    for document in documents:
        wordnet_sim = wordnet_similarity(document["candidates"], document["candidates_pos"], document["tokens"],
                                         document["tokens_pos"])
        document["sim_wordnet"] = wordnet_sim

    with open(wordnet_similarities_path, 'wb') as pkl:
        pickle.dump(documents, pkl)


def load_wordnet_similarities(wordnet_similarities_path):
    with open(wordnet_similarities_path, 'rb') as pkl:
        return pickle.load(pkl)


if __name__ == "__main__":
    labeled_documents_path = '../preprocessing/labeled_documents.json'
    wordnet_similarities_path = 'wordnet_similarities.pickle'
    calculate_score_and_write_pickle(labeled_documents_path, wordnet_similarities_path)
    wordnetsim_documents = load_wordnet_similarities(wordnet_similarities_path)