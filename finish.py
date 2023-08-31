from SingleRank import load_singlerank_scores
from similarity.cosine_similarity import load_cosine_similarities
import json, numpy as np

from main import singlerank_scores_path, cosine_similarities_path, labeled_documents_path


def apply_softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)  # Use the adjusted exponentiated values in both numerator and denominator
    return f_x

def compute_scores(singlerank_documents, cossim_documents, labeled_documents, synonym_lists=[], adjustment_4_sim=2):
    final_documents = labeled_documents[:]

    #metric_scores_cossim = []
    #metric_scores_singlerank = []
    for index, (cossim_document, singlerank_document, final_document) in \
            enumerate(zip(cossim_documents, singlerank_documents, final_documents)):
        similarity_scores = []
        relevance_cossim = []
        relevance_singlerank = []

        #cossim and similarities should ONLY be for candidates, not for other words init?

        index = 0
        for candidate in final_document["candidates"]:
            if isinstance(candidate, list):
                relevance_cossim += [cossim_document["sim_candidates_document"][index]]
                relevance_singlerank += [singlerank_document["singlerank_scores"][index]]
                #HYPERPARAMETER :(
                adjusted_similarity_values_of_can = [x ** adjustment_4_sim for x in cossim_document["sim_candidates_candidates_raw"][index]]
                adjusted_similarity_values_of_can[index] = 0.0 #sim(token_i, token_i) = 0

                #synonym exclusion
                #synonym_list = [(0, 0), (6, 10)]
                #synonyms = [x for x in synonym_list if x[0] == index]
                #if synonyms:
                #    for tupel in synonyms:
                #        adjusted_similarity_values_of_can[tupel[1]] = 0.0

                similarity_scores += [sum(adjusted_similarity_values_of_can)]
                index += 1
            else:
                relevance_cossim += [-1]
                relevance_singlerank += [-1]
                similarity_scores += [-1]

        #metric_score_non_candidates = -1
        #metric_score_candidates = 1/relevance * similarity_of_false_alternatives
        metric_scores_cossim = [
            float(1 / rel_score * sim_score)
            if rel_score != -1 else -1
            for rel_score, sim_score in zip(relevance_cossim, similarity_scores)
        ]
        metric_scores_singlerank = [
            float(1 / rel_score * sim_score)
            if rel_score != -1 else -1
            for rel_score, sim_score in zip(relevance_singlerank, similarity_scores)
        ]

        final_document["candidates_only"] = cossim_document["candidates_only"]

        final_document["similarity_scores"] = similarity_scores
        final_document["relevance_cossim"] = relevance_cossim
        final_document["relevance_singlerank"] = relevance_singlerank
        final_document["metric_scores_cossim"] = metric_scores_cossim
        final_document["metric_scores_singlerank"] = metric_scores_singlerank

        #TODO: SOFTMAX

    return final_documents, metric_scores_cossim


def write_json_final_documents(singlerank_scores_path, cosine_similarities_path, labeled_documents_path, final_documents_path):
    singlerank_documents = load_singlerank_scores(singlerank_scores_path)
    cossim_documents = load_cosine_similarities(cosine_similarities_path)
    documents = json.load(open(labeled_documents_path))
    with open(final_documents_path, 'w') as f:
        json.dump(compute_scores(singlerank_documents, cossim_documents, documents, adjustment_4_sim=2), f)


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


"""



def create_continuous_text_output(final_documents):
    continuous_texts = []
    for document in final_documents:
        continuous_text = document["candidates"]
        for i in range(len(continuous_text)):
            if isinstance(continuous_text[i],list):
                continuous_text[i].append({
                    "S": document["similarity_scores"][i],
                    "R_cossim": document["relevance_cossim"][i],
                    "R_SR": document["relevance_singlerank"][i],
                    "M_cossim": document["metric_scores_cossim"][i],
                    "M_SR": document["metric_scores_singlerank"][i]
                    })
        continuous_texts += [continuous_text]
    return continuous_texts

singlerank_documents = load_singlerank_scores(singlerank_scores_path)
cossim_documents = load_cosine_similarities(cosine_similarities_path)
labeled_documents = json.load(open(labeled_documents_path))
final_documents, m_scores = compute_scores(singlerank_documents, cossim_documents, labeled_documents, adjustment_4_sim=2)

continuous_texts = create_continuous_text_output(final_documents)
