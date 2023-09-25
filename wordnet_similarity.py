import json
import pickle
import nltk
from nltk.corpus import wordnet
import numpy as np
from itertools import chain


def meronyms_of_same_helonym_counter(candidates, candidates_pos, tokens, tokens_pos):
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def map_pos(pos):
        pos = pos[0].lower()
        if pos not in ["n", "v", "a", "r", "s"]:
            pos = "n"
        return pos

    candidates_pos = list(map(map_pos, candidates_pos))
    tokens_lemmatized = [lemmatizer.lemmatize(token, map_pos(token_pos[1])) for (token, token_pos) in zip(tokens, tokens_pos)]
    scores = []
    i = 0
    print(candidates_pos, tokens_lemmatized, [candidate for candidate in candidates if isinstance(candidate, list)])
    for candidate in [candidate for candidate in candidates if isinstance(candidate, list)]:
        pos = candidates_pos[i:i+len(candidate)]
        i += len(candidate)
        #take last noun or last wordy
        print(candidate, pos)
        if len(candidate) > 1:
            if "n" in pos:
                indexes = [index for index, value in enumerate(pos) if value == "n"]
                candidate = [candidate[indexes[-1]]]
                pos = ["n"]
            else:
                candidate = candidate[-1]
                pos = [pos[-1]]
        print(candidate, pos)

        synsets = wordnet.synsets(lemmatizer.lemmatize(candidate[0], pos[0]))
        synonyms = set(chain.from_iterable(
            [word.lemma_names() for word in synsets]))
        holonyms = [holonym for synset in synsets for holonym in synset.member_holonyms()]
        hypernyms = [hypernym for synset in synsets for hypernym in synset.hypernyms()]

        #SAME FOR HYPERNYM
        # You can also use part_holonyms() for parts of a whole
        #Instances (specific persons, countries and geographic entities)
        #(((troponyms for verbs)))



        def check_for_others(superordinates, subordinate_function, tokens_lemmatized, candidate):
            score = 0
            # score +=1 if meronym that is no synonym and not the candidate itself
            for superordinate in superordinates:
                subordinates = set(chain.from_iterable(
                    [subordinate.lemma_names() for subordinate in eval(subordinate_function)]))
                subordinates = [subordinate.lower() for subordinate in subordinates if subordinate not in synonyms]
                for token in tokens_lemmatized:
                    if token.lower() in subordinates and token.lower() != candidate.lower():
                        score += 1
            return score
        scores += [check_for_others(holonyms, "superordinate.member_meronyms()",tokens_lemmatized, candidate)]
        scores += [check_for_others(hypernyms, "superordinate.hyponyms()",tokens_lemmatized, candidate)]

        """
                score = 0
                for holonym in holonyms:  # You can also use part_holonyms() for parts of a whole
                    meronyms = set(chain.from_iterable(
                        [meronym.lemma_names() for synset in synsets for meronym in holonym.member_meronyms()]))
                    # score +=1 if meronym that is no synonym and not the candidate itself
                    meronyms = [meronym.lower() for meronym in meronyms if meronym not in synonyms]
                    for token in tokens_lemmatized:
                        if token.lower() in meronyms and not candidate:
                            score += 1
                scores += [score]
                """
    return scores
"""
        score = 0
        for synonym in synsets:
            if synonym in tokens_lemmatized:
                score += 1
        scores += [score]
"""


def calculate_score_and_write_pickle(labeled_documents_path, wordnet_similarities_path):
    documents_json = open(labeled_documents_path)
    documents = json.load(documents_json)

    for document in documents:
        wordnet_sim = meronyms_of_same_helonym_counter(
            document["candidates"], document["candidates_pos"], document["tokens"], document["tokens_pos"])
        document["sim_wordnet"] = wordnet_sim

    with open(wordnet_similarities_path, 'wb') as pkl:
        pickle.dump(documents, pkl)


def load_wordnet_similarities(wordnet_similarities_path):
    with open(wordnet_similarities_path, 'rb') as pkl:
        return pickle.load(pkl)


if __name__ == "__main__":
    labeled_documents_path = 'preprocessing/labeled_documents.json'
    wordnet_similarities_path = 'similarity/wordnet_similarities.pickle'
    calculate_score_and_write_pickle(labeled_documents_path, wordnet_similarities_path)
    wordnetsim_documents = load_wordnet_similarities(wordnet_similarities_path)