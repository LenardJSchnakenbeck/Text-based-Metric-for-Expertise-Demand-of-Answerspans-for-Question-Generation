import pandas as pd

from SingleRank import load_singlerank_scores
from similarity.cosine_similarity import load_cosine_similarities
import json, copy, numpy as np
from sklearn.preprocessing import RobustScaler #, MinMaxScaler


from main import singlerank_scores_path, cosine_similarities_path, labeled_documents_path


def apply_softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)  # Use the adjusted exponentiated values in both numerator and denominator
    return f_x


def apply_min_max_normalization(values):
    min_val = min(values)
    max_val = max(values)

    # Check if all values are the same to avoid division by zero
    if min_val == max_val:
        return [0.0] * len(values)

    return [(value - min_val) / (max_val - min_val) for value in values]


def apply_robust_normalization(data):
    # This method is less sensitive to outliers.
    data = np.array(data).reshape(-1, 1)
    rs = RobustScaler().fit(data)
    return rs.transform(data).tolist()


def compute_scores(singlerank_documents, cossim_documents, labeled_documents, synonym_lists=[], adjustment_4_sim=2):
    final_documents = labeled_documents[:]

    #metric_scores_cossim = []
    #metric_scores_singlerank = []
    for index, (cossim_document, singlerank_document, final_document) in \
            enumerate(zip(cossim_documents, singlerank_documents, final_documents)):
        similarity_scores = []
        relevance_cossim = []
        relevance_singlerank = []

        index = 0
        for candidate in final_document["candidates"]:
            if isinstance(candidate, list):
                relevance_cossim += [cossim_document["sim_candidates_document"][index]]
                relevance_singlerank += [singlerank_document["singlerank_scores"][index]]
                adjusted_similarity_values_of_can = [x for x in cossim_document["sim_candidates_candidates_raw"][index]]
                adjusted_similarity_values_of_can[index] = 0.0 #sim(token_i, token_i) = 0

                #synonym exclusion
                #synonym_list = [(0, 0), (6, 10)]
                #synonyms = [x for x in synonym_list if x[0] == index]
                #if synonyms:
                #    for tupel in synonyms:
                #        adjusted_similarity_values_of_can[tupel[1]] = 0.0

                #HYPERPARAMETER :(
                similarity_scores += [sum([x for x in adjusted_similarity_values_of_can if x >= 0.3])]
                index += 1
            ##else:
                #relevance_cossim += [-1]
                #relevance_singlerank += [-1]
                #similarity_scores += [-1]

        #min-max normalization (range: 0-1)
        relevance_cossim = apply_min_max_normalization(relevance_cossim)
        similarity_scores = apply_min_max_normalization(similarity_scores)

        #metric_score_non_candidates = -1
        #metric_score_candidates = 1/relevance * similarity_of_false_alternatives
        metric_scores_cossim = [
            float((1 - rel_score) * sim_score)
            for rel_score, sim_score in zip(relevance_cossim, similarity_scores)
            if rel_score != -1
        ]
        metric_scores_singlerank = [
            float((1 - rel_score) * sim_score)
            for rel_score, sim_score in zip(relevance_singlerank, similarity_scores)
            if rel_score != -1
        ]

        final_document["candidates_only"] = cossim_document["candidates_only"]

        final_document["similarity_scores"] = similarity_scores
        final_document["relevance_cossim"] = relevance_cossim
        final_document["relevance_singlerank"] = relevance_singlerank
        final_document["metric_scores_cossim"] = metric_scores_cossim
        final_document["metric_scores_singlerank"] = metric_scores_singlerank

        #TODO: Boxplots für unterschiedliche Values
        #TODO: Texte raussuchen
        #TODO: Synonym

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

I can implement 3., there should be a way to find synonyms, however

 
 
"""



def create_continuous_text_output(final_documents):
    continuous_texts = []
    for document in final_documents:
        #continuous_text = document["candidates"][:]
        continuous_text = copy.deepcopy(document["candidates"])
        offset = 0
        for i in range(len(continuous_text)):
            if isinstance(continuous_text[i],list):
                continuous_text[i].append({
                    "S": round(document["similarity_scores"][i - offset], 3),
                    "R_cos": round(document["relevance_cossim"][i - offset], 3),
                    "R_SR": round(document["relevance_singlerank"][i - offset], 3),
                    "M_cos": round(document["metric_scores_cossim"][i - offset], 3),
                    #"MR_cos": document["metric_scores_cossim_rank"][i],
                    "M_SR": round(document["metric_scores_singlerank"][i - offset], 3),
                    #"MR_SR": document['metric_scores_singlerank_rank'][i]
                    })
            else:
                offset += 1
        continuous_texts += [continuous_text]
    return continuous_texts

singlerank_documents = load_singlerank_scores(singlerank_scores_path)
cossim_documents = load_cosine_similarities(cosine_similarities_path)
labeled_documents = json.load(open(labeled_documents_path))
final_documents, m_scores = compute_scores(singlerank_documents, cossim_documents, labeled_documents, adjustment_4_sim=2)

continuous_texts = create_continuous_text_output(final_documents)


def create_candidates_df(final_document):
    keys = [
        "candidates_only",
        "similarity_scores",
        "relevance_cossim",
        "relevance_singlerank",
        "metric_scores_cossim",
        "metric_scores_singlerank"
    ]
    data = [final_document[key] for key in keys]
    df = pd.DataFrame(data)
    df_transposed = df.transpose()
    df_transposed.columns = keys
    df_transposed.insert(0, "index", range(len(df_transposed["candidates_only"])))
    return df_transposed
    #df_transposed.sort_values("metric_scores_cossim")

def create_rank_columns(df, sort_by=["metric_scores_cossim", "metric_scores_singlerank"]):
    sorted_df = df
    for column in sort_by:
        #df.sort_values("metric_scores_cossim")
        sorted_df = sorted_df.sort_values(by=column)
        sorted_df[str(column + "_rank")] = range(len(sorted_df))
    return sorted_df.sort_values("index")

import scipy
def compare_metric_scores(df):
    return scipy.stats.kendalltau(df['metric_scores_cossim_rank'], df['metric_scores_singlerank_rank'])


def create_html_hover_text(text_with_values, html_doc_name="html_doc.html", additional_text=""):
    html_template_1 = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Define styles for the tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: blue;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 150px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -75px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        .text {
            margin: 50px;
            margin-top: 300px;
            font-size: x-large;
        }
    </style>
</head>
<body>
    <p class="text"> 
    """
    html_template_2 = """
    </p>
</body>
</html>
"""
    text = ""
    for i in text_with_values:
        if isinstance(i, list):
            text += str(' <span class="tooltip"> '+' '.join(i[0:-1])+' <span class="tooltiptext"> '+repr(i[-1])+' </span></span> ')
        else:
            text += str(" "+i+" ")
    html_output = html_template_1 + text + "<br><br>" + additional_text + html_template_2

    with open(html_doc_name, "w") as f:
        f.write(html_output)

    return html_output


i = 0
for final_document in final_documents:
    df = create_candidates_df(final_document)
    create_html_hover_text(
        continuous_texts[i],
        str("html_doc " + final_document["title"]),
        str([(name, round(result, 5)) for name, result in zip(
                ["stat: ", "Pvalue: "],
                scipy.stats.kendalltau(df['metric_scores_cossim'], df['metric_scores_singlerank']))
        ])
    )
    i += 1
