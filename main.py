from preprocessing import candidate_phrases
from similarity import cosine_similarity, wordnet_similarity
from metric_creation import compute_scores, write_json_final_documents, load_final_documents
import SingleRank
import json


#TODO: bei Candidates index (Stelle in tokens[]) hinzufügen
#TODO: Gute Ausgabe mit Werten
#TODO: bei similarities synonyme raus nehmen #Entity Coreference labelling
#TODO: similarities of false alternatives zusammen zählen

#path_to_raw_wikipedia_texts = "../../downloads/raw.txt" #https://github.com/tscheepers/Wikipedia-Summary-Dataset
#source_documents_path = 'preprocessing/wikipedia_texts.json'
source_documents_path = "experiment_texts.json"
labeled_documents_path = 'preprocessing/labeled_documents.json'
embeddings_path = 'similarity/embeddings.pickle'
cosine_similarities_path = 'similarity/cosine_similarities.pickle'
wordnet_similarities_path = 'similarity/wordnet_similarities.pickle'
singlerank_scores_path = 'similarity/singlerank_scores.pickle'
final_documents_path = 'final_documents.json'

#source_documents_path = 'preprocessing/wikipedia_texts_DiY.json'


if __name__ == "__main__" and input("recalculate everything?") == "y":
    #create sjson from raw wikipedia texts
    #candidate_phrases.load_wikipedia_and_create_json(path_to_raw_wikipedia_texts, source_documents_path)

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
    labeled_documents = json.load(open(labeled_documents_path))
    singlerank_documents = SingleRank.load_singlerank_scores(singlerank_scores_path)
    cossim_documents = cosine_similarity.load_cosine_similarities(cosine_similarities_path)
    wordnet_documents = wordnet_similarity.load_wordnet_similarities(wordnet_similarities_path)
    documents = compute_scores(
        singlerank_documents, cossim_documents, wordnet_documents, labeled_documents)
    write_json_final_documents(documents, final_documents_path)
    print("results are loaded :)")

else:
    documents = load_final_documents(final_documents_path)



