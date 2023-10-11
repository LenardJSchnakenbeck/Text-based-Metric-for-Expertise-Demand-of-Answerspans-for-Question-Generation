import pandas as pd
import json, numpy as np
from sklearn.preprocessing import RobustScaler #, MinMaxScaler


def apply_softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)  # Use the adjusted exponentiated values in both numerator and denominator
    return f_x


def apply_robust_normalization(data):
    # This method is less sensitive to outliers.
    data = np.array(data).reshape(-1, 1)
    rs = RobustScaler().fit(data)
    return rs.transform(data).tolist()


def apply_min_max_normalization(values):
    min_val = min(values)
    max_val = max(values)

    # Check if all values are the same to avoid division by zero
    if min_val == max_val:
        return [0.0] * len(values)

    return [(value - min_val) / (max_val - min_val) for value in values]


def apply_custom_min_max_scaling(data, min_value=0.1, max_value=1):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [((x - min_val) / (max_val - min_val)) * (max_value - min_value) + min_value for x in data]
    return scaled_data


def compute_scores(singlerank_documents, cossim_documents, wordnet_documents, labeled_documents):
    final_documents = labeled_documents[:]

    #metric_scores_cossim = []
    #metric_scores_singlerank = []
    for index, (cossim_document, wordnet_document, singlerank_document, final_document) in \
            enumerate(zip(cossim_documents, wordnet_documents, singlerank_documents, final_documents)):
        similarity_cossim = []
        similarity_wordnet = []
        relevance_cossim = []
        relevance_singlerank = []

        index = 0
        for candidate in final_document["candidates"]:
            if isinstance(candidate, list):
                relevance_cossim += [cossim_document["sim_candidates_document"][index]]
                relevance_singlerank += [singlerank_document["singlerank_scores"][index]]
                similarity_wordnet += [wordnet_document["sim_wordnet"][index]]
                similarity_cossim_list = cossim_document["sim_candidates_candidates_raw"][index]
                similarity_cossim_list[index] = 0.0 #sim(token_i, token_i) = 0

                #HYPERPARAMETER :(
                similarity_cossim += [sum(similarity_cossim_list)]
                index += 1


        def calculate_metric(similarity_scores, relevance_scores):
            similarity_scores = apply_custom_min_max_scaling(similarity_scores)
            relevance_scores = apply_custom_min_max_scaling(relevance_scores)
            metric_scores = [
                float((1 - rel_score) * sim_score)
                for rel_score, sim_score in zip(relevance_scores, similarity_scores)
                if rel_score != -1
            ]
            return metric_scores

        final_document["candidates_only"] = cossim_document["candidates_only"]

        final_document["similarity_cossim"] = similarity_cossim
        final_document["similarity_wordnet"] = similarity_wordnet
        final_document["relevance_cossim"] = relevance_cossim
        final_document["relevance_singlerank"] = relevance_singlerank

        final_document["m_s_cos_r_cos"] = calculate_metric(similarity_cossim, relevance_cossim)
        final_document["m_s_cos_r_sr"] = calculate_metric(similarity_cossim, relevance_singlerank)
        final_document["m_s_wn_r_cos"] = calculate_metric(similarity_wordnet, relevance_cossim)
        final_document["m_s_wn_r_sr"] = calculate_metric(similarity_wordnet, relevance_singlerank)


        #TODO: Boxplots für unterschiedliche Values
        #TODO: Texte raussuchen

    return final_documents


def write_json_final_documents(final_documents, final_documents_path):
    with open(final_documents_path, 'w') as f:
        json.dump(final_documents, f)


def load_final_documents(final_documents_path):
    with open(final_documents_path, 'rb') as json_file:
        return json.load(json_file)


if __name__ == "__main__":
    """
    from main import singlerank_scores_path, cosine_similarities_path, \
        labeled_documents_path, wordnet_similarities_path, final_documents_path
    from SingleRank import load_singlerank_scores
    from similarity.cosine_similarity import load_cosine_similarities
    from similarity.wordnet_similarity import load_wordnet_similarities

    singlerank_documents = load_singlerank_scores(singlerank_scores_path)
    cossim_documents = load_cosine_similarities(cosine_similarities_path)
    labeled_documents = json.load(open(labeled_documents_path))
    wordnet_documents = load_wordnet_similarities(wordnet_similarities_path)
    final_documents = compute_scores(
        singlerank_documents, cossim_documents, wordnet_documents, labeled_documents)
    """
    from main import final_documents_path
    documents = load_final_documents(final_documents_path)


"""

candidates_no = 0
l = []
for i in range(len(cossim_documents[0]["sim_candidates_candidates_raw"][candidates_no])):
    l+= [(i, cossim_documents[0]["candidates_only"][i], cossim_documents[0]["sim_candidates_candidates_raw"][candidates_no][i])]
    
l = sorted(l, key=lambda x: x[2])



m_s=[]
for i in final_documents[2]["metric_scores_cossim"]:
    if i != -1:
        m_s += [i]
        
len(m_s)
c_m=[]
for a,b in zip(m_s, final_documents[2]["candidates_only"]):
    c_m += [[a,b]]

#MIN-MAX SCALING DU HUND


sorted_arr = sorted(arr, key=custom_sort, reverse=True) 
sorted_arr = sorted(arr, key=lambda a: a[1], reverse=True) 

key=lambda student: student[2]
"""


"""
Similarity: nur wirklich ähnliche Phrasen zählen (nicht viele etwas ähnliche)
-> Threshold: Nur Werte oberhalb des Grenzwertes einbeziehen
    Höhe des Grenzwertes: 0.3

    TODOs:
    - Metrik Berechnung aktualisieren
    - darüber schreiben
---------------------------
Ich brauche nicht similarity, sondern ob entität aus gleicher domain ist
Venus, Mars, Jupiter sind entitäten aus gleicher domain
Planet, Astronomie, Jupiter sind keine entitäten aus gleicher domain

Bei wordnet schauen? -same semantic category-
Juptier(member holonym: solar system)
    solar system(member meronym: Venus, Mars, Jupiter)

Autism(domain category: psychiatry, psychopathology, psychological medicine)
    psychiatry, psychopathology, psychological medicine (domain term category: autism..)
    
1. lemmatise word
    1.1. get pos 
2. find related words
3. search in text (not candidates) for related words (lemmatized)
 
 Alles manuell machen du hurenshon kriegst nichts hin, diese arbeit ist ein bösartiges spiel mit deinem leben
 Approaches beschreiben und sagen dass die alle komplett fürn arsch sind keine zeit für hyperparameter search 
 nichts halbes und erst recht nicht fucking ganzes
 

dear Irene,

unfortunately, I discovered some problems having a first look at the metric.
It concerns the automatic analysis of false alternatives to an answer span, where I try to find similar entities in the
text which could be confused with the answerspan. E.g. if Jupiter is the answerspan, venus should be a false alternative.
My first approach was calculating the similarity using Bert-Embeddings and cosine-similarity.
There are three implementation approaches:

1. summing the similarity values for every word in the text
Problem: many quite similar words (Astronomy, solar system, etc) have the same influence as one very similar word (Venus)
But (Astronomy, solar system, etc) can't be confused with Jupiter

2. find hyperparameters: 
a threshold (so only very similar words are considered) 
or a function that emphasizes high values and lowers low values.
Problem: Hyperparameter-search would go beyond the scope of the work

3. Wordnet
Use wordnet to find entities from the same semantic category by finding holonyms and their meronyms,
e.g. parent (holonym: family unit), family unit (meronyms: child/kid, parent, sibling/sib)
however, this is somewhat limited, since father or mother is not included.


a. qualitative (pre-test): kick-out singlerank-relevance and cossim-similarity?
b. quantitative study gets human values, compare with metric values
 
 
 
 #eines Schwierigkeitsindexes für generierte Fragen
 #Evaluation einer Metrik zur Einschätzung der Expertisenanforderung von Phrasen in Texten
 
 zweitprüfer: pauli (josef.pauli@uni-due.de)
 https://campus.uni-due.de/lsf/rds?state=wtree&search=1&trex=step&root120232=356121%7C355019%7C355710%7C356409%7C355456%7C353779&P.vx=kurz
 
"""


