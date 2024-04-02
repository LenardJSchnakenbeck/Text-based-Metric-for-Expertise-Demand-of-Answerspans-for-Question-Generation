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


def apply_laplace_smoothing(values, smoothing_parameter=0.1):
    number_values = len(values)
    smoothed_values = []
    for value in values:
        smoothed_values += [
            (value + smoothing_parameter) /
            (sum(values) + smoothing_parameter * number_values)
        ]
    return smoothed_values

def apply_smoothed_normalization(values):
    values = [value + 1 for value in values]
    values += [0]
    values = apply_min_max_normalization(values)
    return values[:-1]

def apply_custom_min_max_scaling(data, min_value=0.01, max_value=1):
    if min(data) != max(data):
        scaled_data = [((x - min(data)) / (max(data) - min(data))) * (max_value - min_value) + min_value for x in data]
        return scaled_data
    else:
        return data



def compute_scores(singlerank_documents, singlerank_meaned_documents, cossim_documents, wordnet_documents, labeled_documents):
    final_documents = labeled_documents[:]

    for index, (cossim_document, wordnet_document, singlerank_document, singlerank_meaned_document, final_document) in \
            enumerate(zip(cossim_documents, wordnet_documents, singlerank_documents, singlerank_meaned_documents, final_documents)):
        similarity_cossim = []
        similarity_wordnet = []
        relevance_cossim = []
        relevance_singlerank = []
        relevance_singlerank_meaned = []

        index = 0
        for candidate in final_document["candidates"]:
            if isinstance(candidate, list):
                relevance_cossim += [cossim_document["sim_candidates_document"][index]]
                relevance_singlerank += [singlerank_document["singlerank_scores"][index]]
                relevance_singlerank_meaned += [singlerank_meaned_document["singlerank_scores"][index]]
                similarity_wordnet += [wordnet_document["sim_wordnet"][index]]
                similarity_cossim_list = cossim_document["sim_candidates_candidates_raw"][index]
                similarity_cossim_list[index] = 0.0 #sim(token_i, token_i) = 0

                similarity_cossim += [sum(similarity_cossim_list)]
                index += 1


        def calculate_metric(similarity_scores, relevance_scores):
            metric_scores = [
                float((1 - rel_score) * sim_score)
                for rel_score, sim_score in zip(relevance_scores, similarity_scores)
                if rel_score != -1
            ]
            return metric_scores

        final_document["candidates_only"] = cossim_document["candidates_only"]

        final_document["similarity_cossim"] = similarity_cossim
        final_document["similarity_wordnet"] = similarity_wordnet
        smoothed_similarity_wordnet = apply_laplace_smoothing(similarity_wordnet)
        final_document["similarity_wordnet_smoothed"] = smoothed_similarity_wordnet
        final_document["relevance_cossim"] = relevance_cossim
        final_document["relevance_singlerank"] = relevance_singlerank
        final_document["relevance_singlerank_meaned"] = relevance_singlerank_meaned

        final_document["CosineSim_CosineRel"] = calculate_metric(
            apply_min_max_normalization(similarity_cossim),
            apply_min_max_normalization(relevance_cossim))
        final_document["CosineSim_SinglerankRel"] = calculate_metric(
            apply_min_max_normalization(similarity_cossim),
            apply_min_max_normalization(relevance_singlerank))

        final_document["WordnetSim_CosineRel"] = calculate_metric(
            apply_smoothed_normalization(similarity_wordnet), #smoothed
            apply_min_max_normalization(relevance_cossim))
        final_document["WordnetSim_SinglerankRel"] = calculate_metric(
            apply_smoothed_normalization(similarity_wordnet), #smoothed
            apply_min_max_normalization(relevance_singlerank))

    #SingleRank_meaned
        final_document["CosineSim_Singlerank_meanedRel"] = calculate_metric(
            apply_min_max_normalization(similarity_cossim),
            apply_min_max_normalization(relevance_singlerank_meaned))
        final_document["WordnetSim_Singlerank_meanedRel"] = calculate_metric(
            apply_smoothed_normalization(similarity_wordnet), #smoothed
            apply_min_max_normalization(relevance_singlerank_meaned))

    return final_documents


def write_json_final_documents(final_documents, final_documents_path):
    with open(final_documents_path, 'w') as f:
        json.dump(final_documents, f)


def load_final_documents(final_documents_path):
    with open(final_documents_path, 'rb') as json_file:
        return json.load(json_file)


if __name__ == "__main__":
    from main import final_documents_path
    documents = load_final_documents(final_documents_path)


