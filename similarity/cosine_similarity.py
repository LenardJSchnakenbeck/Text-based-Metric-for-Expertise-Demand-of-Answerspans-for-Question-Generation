import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

from preprocessing.candidate_phrases import get_main_word, map_pos_tokenizer_to_lemmatizer

from itertools import chain
import nltk
from nltk.corpus import wordnet
lemmatizer = nltk.stem.WordNetLemmatizer()

def encode_text_and_avg_candidates(documents):
    model = SentenceTransformer('all-mpnet-base-v2')
    result = []
    for document in documents:
        text_encoded = model.encode(document['text'])
        candidates = document['candidates']
        candidates_encoded = model.encode([" ".join(can) if isinstance(can, list) else can for can in candidates])
        result.append({
            'candidates': candidates_encoded,   # candidate embeddings
            'text': text_encoded                # document embedding
        })
    return result

def write_pickle_encoded_documents(embeddings_path,labeled_documents_path):
    documents_json = open(labeled_documents_path)
    documents = json.load(documents_json)
    with open(embeddings_path, 'wb') as pkl:
        pickle.dump(encode_text_and_avg_candidates(documents), pkl)

def cossim_candidates_document(embeddings_path):
    with open(embeddings_path, 'rb') as pkl:
        document_embeddings = pickle.load(pkl)

    result = []
    for document_embedding in document_embeddings:
        text_embedding = document_embedding['text']
        candidates_document_similarites = []
        for candidate in document_embedding['candidates']:
            candidates_document_similarites.append(
                float(cosine_similarity(candidate.reshape(1,-1), text_embedding.reshape(1,-1))[0])
            )
        result.append(candidates_document_similarites)
    return result

def cossim_candidates_candidates(embeddings_path, documents, synonyms=[(['C.E.Gates'],['Gates']),(['Gates'],['C.E.Gates'])]):
    with open(embeddings_path, 'rb') as pkl:
        document_embeddings = pickle.load(pkl)

    result = []
    for document_embedding, document in zip(document_embeddings, documents):
        #for every candidate (1) calculate cossim to every candidate (2)
        candidates1 = []
        i = 0
        for (candidate1emb, candidate1) in zip(document_embedding['candidates'], document['candidates']):
            candidates2 = []
            #calculate similarity to every other word except synonyms
            for j, (candidate2emb, candidate2) in enumerate(zip(document_embedding['candidates'], document['candidates'])):
                if candidate1 == ["Gates"]: print(candidate1,candidate2)
                if (candidate1,candidate2) in synonyms:
                    candidates2.append(0.0)
                elif isinstance(candidate2, list):
                    candidate2, _ = get_main_word(candidate2, document["tokens_pos"][j])
                else:
                    candidates2.append(float(
                        cosine_similarity(candidate1emb.reshape(1, -1), candidate2emb.reshape(1, -1))[0]))
            candidates1.append(candidates2)
            i += 1
        result.append(candidates1)
    return result


def calculate_and_write_pickle_cossim(labeled_documents_path, embeddings_path,cosine_similarities_path, apply_softmax = False):
    documents_json = open(labeled_documents_path)
    documents = json.load(documents_json)

    #candidates-document cosine-similarity
    cosine_values_can_doc = cossim_candidates_document(embeddings_path)
    if apply_softmax:
        cosine_values_can_doc = np.exp(np.array(cosine_values_can_doc)) / np.sum(np.exp(np.array(cosine_values_can_doc)))

    for index, document in enumerate(documents):
        document['sim_candidates_document'] = cosine_values_can_doc[index]
        document["candidates_only"] = [candidate for candidate in document["candidates"] if isinstance(candidate, list)]

    #candidates-candidates cosine-similarity
    cosine_values_can_can = cossim_candidates_candidates(embeddings_path, documents)

    for index, document in enumerate(documents):
        document['sim_candidates_candidates_raw'] = cosine_values_can_can[index]
    #    document['sim_candidates_candidates'] = sum(cosine_values_can_can[index])

    with open(cosine_similarities_path, 'wb') as pkl:
        pickle.dump(documents, pkl)


def load_cosine_similarities(cosine_similarities_path):
    with open(cosine_similarities_path, 'rb') as pkl:
        return pickle.load(pkl)


#encode_text_and_avg_candidates(documents)  #1. encode and serialize
#calculate_and_update_json_candidates_document_cossim() #2. calculate and serialize
#load_texts() #3. load all documents

if __name__ == "__main__":
    labeled_documents_path = '../preprocessing/labeled_wikipedia_texts.json'
    embeddings_path = "embeddings.pickle"
    cosine_similarities_path = "cosine_similarities.pickle"
    write_pickle_encoded_documents(embeddings_path, labeled_documents_path)
    calculate_and_write_pickle_cossim(embeddings_path, cosine_similarities_path)
    cossim_documents = load_cosine_similarities(cosine_similarities_path)