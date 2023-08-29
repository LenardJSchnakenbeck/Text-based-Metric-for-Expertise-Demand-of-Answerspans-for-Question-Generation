from preprocessing import candidate_phrases
from similarity import cosine_similarity
import SingleRank
import numpy as np
import json


def load_wikipedia_and_create_json(path_to_raw_wikipedia_texts, source_documents_path, number_of_lines=10000):
    #load
    with open(path_to_raw_wikipedia_texts, "r") as f:
        if number_of_lines:
            texts = f.readlines(number_of_lines)
        else:
            texts = f.readlines()
    texts.pop(-1)
    #refactor
    texts_dicts = []
    for text in texts:
        separator = text.index(" ||| ")
        title = text[:separator]
        texts_dicts += [{"title": title, "text": text[separator + 5:]}]
    #save
    with open(source_documents_path, "w") as f:
        json.dump(texts_dicts, f)

#TODO: bei Candidates index (Stelle in tokens[]) hinzufügen
#TODO: Gute Ausgabe mit Werten
#TODO: bei similarities synonyme raus nehmen
#TODO: similarities of false alternatives zusammen zählen

path_to_raw_wikipedia_texts = "../../downloads/raw.txt"
source_documents_path = 'preprocessing/wikipedia_texts.json'
labeled_documents_path = 'preprocessing/labeled_documents.json'
embeddings_path = 'similarity/embeddings.pickle'
cosine_similarities_path = 'similarity/cosine_similarities.pickle'
singlerank_scores_path = 'similarity/singlerank_scores.pickle'

#create sjson from raw wikipedia texts
load_wikipedia_and_create_json(path_to_raw_wikipedia_texts, source_documents_path)

#chunk candidates by pos-tag
candidate_phrases.write_json_labeled_documents(source_documents_path, labeled_documents_path)
print("candidates chunked!")

#calculate similarities (candidates-candidates & candidates-documents)
cosine_similarity.write_pickle_encoded_documents(embeddings_path, labeled_documents_path)
cosine_similarity.calculate_and_write_pickle_cossim(labeled_documents_path, embeddings_path, cosine_similarities_path)
print("cosine similarities calculated!")

#calculate SingleRank scores
SingleRank.calculate_and_write_pickle_singlerank_scores(labeled_documents_path, singlerank_scores_path)
print("singlerank applied!")

#load results
singlerank_documents = SingleRank.load_singlerank_scores(singlerank_scores_path)
cossim_documents = cosine_similarity.load_cosine_similarities(cosine_similarities_path)
documents = json.load(open(labeled_documents_path))
print("results are loaded :)")

##RELEVANCY
#KeyBERT-Approach: Similarities between candidates and documents
#SingleRank

##SIMILARITY OF FALSE ALTERNATIVES
#Similarities between candidates
    #apply Sigmoid for Influence control





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
