import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

documents_json = open('../preprocessing/labeled_wikipedia_texts.json')
documents = json.load(documents_json)




def encode_text_and_nounphrases(documents):
    model = SentenceTransformer('all-mpnet-base-v2')
    result = []
    for document in documents:
        nounphrases_encoded = []
        for nounphrase in document['nounphrases']:
            nounphrases_encoded.append(model.encode(nounphrase))
        text_encoded = model.encode(document['text'])
        result.append({
            'nounphrases': nounphrases_encoded,
            'text': text_encoded
        })
    return result

def encode_documents():
    with open('embeddings.pickle', 'wb') as pkl:
        pickle.dump(encode_text_and_nounphrases(documents), pkl)


def cosine():
    with open('embeddings.pickle', 'rb') as pkl:
        document_embeddings = pickle.load(pkl)

    result = []
    for document in document_embeddings:
        result.append(cosine_similarity(document['nounphrases'], [document['text']]))
    return result


def calculate_and_save_bert_cossim():
    cosine_values = [[x[0] for x in y] for y in cosine()]

    for index, document in enumerate(documents):
        document['bert_sim'] = cosine_values[index]


    with open('texts_bert_cossim.pickle', 'wb') as pkl:
        pickle.dump(documents, pkl)

def load_texts():
    with open('texts_bert_cossim.pickle', 'rb') as pkl:
        return pickle.load(pkl)

#encode_documents() #1. encode and serialize
#calculate_and_save_bert_cossim() #2. calculate and serialize
#load_texts() #3. load all documents




