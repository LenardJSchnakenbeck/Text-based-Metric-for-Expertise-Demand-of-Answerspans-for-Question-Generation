from preprocessing import candidate_phrases
from similarity import cosine_similarity, wordnet_similarity
from metric_creation import compute_scores, write_json_final_documents, load_final_documents
import SingleRank
import json

source_documents_path = "experiment_texts.json"
labeled_documents_path = 'preprocessing/labeled_documents.json'
embeddings_path = 'similarity/embeddings.pickle'
cosine_similarities_path = 'similarity/cosine_similarities.pickle'
wordnet_similarities_path = 'similarity/wordnet_similarities.pickle'
singlerank_scores_path = 'similarity/singlerank_scores.pickle'
singlerank_meaned_scores_path = 'similarity/singlerank_meaned_scores.pickle' #
final_documents_path = 'final_documents.json'



if __name__ == "__main__" and input("recalculate everything?").lower() in ["y", "yes", "ja"]:

    candidate_phrases.write_study_texts(labeled_documents_path)

    cosine_similarity.write_pickle_encoded_documents(embeddings_path, labeled_documents_path)
    cosine_similarity.calculate_and_write_pickle_cossim(labeled_documents_path, embeddings_path, cosine_similarities_path)
    print("cosine similarities calculated!")
    wordnet_similarity.calculate_score_and_write_pickle(labeled_documents_path, wordnet_similarities_path)


    SingleRank.calculate_and_write_pickle_singlerank_scores(labeled_documents_path, singlerank_meaned_scores_path)
    print("singlerank applied!")

    labeled_documents = json.load(open(labeled_documents_path))
    singlerank_documents = SingleRank.load_singlerank_scores(singlerank_scores_path)
    singlerank_meaned_documents = SingleRank.load_singlerank_scores(singlerank_meaned_scores_path)
    cossim_documents = cosine_similarity.load_cosine_similarities(cosine_similarities_path)
    wordnet_documents = wordnet_similarity.load_wordnet_similarities(wordnet_similarities_path)
    documents = compute_scores(
        singlerank_documents, singlerank_meaned_documents, cossim_documents, wordnet_documents, labeled_documents)
    write_json_final_documents(documents, final_documents_path)
    print("results are loaded :)")

else:
    import pandas as pd

    documents = load_final_documents(final_documents_path)

   metric_results = pd.DataFrame({
        'CosineSim_CosineRel': documents[0]['CosineSim_CosineRel'],
        'WordnetSim_SinglerankRel': documents[0]['WordnetSim_SinglerankRel'],
        'WordnetSim_CosineRel': documents[0]['WordnetSim_CosineRel'],
        'CosineSim_SinglerankRel': documents[0]['CosineSim_SinglerankRel'],
        'CosineRel': documents[0]['relevance_cossim'],
        "SinglerankRel": documents[0]['relevance_singlerank'],
        'WordnetSim': documents[0]['similarity_wordnet'],
        'CosineSim': documents[0]['similarity_cossim']
    })
