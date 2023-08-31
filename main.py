from preprocessing import candidate_phrases
from similarity import cosine_similarity
import SingleRank
import json


#TODO: bei Candidates index (Stelle in tokens[]) hinzufügen
#TODO: Gute Ausgabe mit Werten
#TODO: bei similarities synonyme raus nehmen #Entity Coreference labelling
#TODO: similarities of false alternatives zusammen zählen

path_to_raw_wikipedia_texts = "../../downloads/raw.txt"
source_documents_path = 'preprocessing/wikipedia_texts.json'
labeled_documents_path = 'preprocessing/labeled_documents.json'
embeddings_path = 'similarity/embeddings.pickle'
cosine_similarities_path = 'similarity/cosine_similarities.pickle'
singlerank_scores_path = 'similarity/singlerank_scores.pickle'

source_documents_path = 'preprocessing/wikipedia_texts_DiY.json'


if __name__ == "__main__":
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
    singlerank_documents = SingleRank.load_singlerank_scores(singlerank_scores_path)
    cossim_documents = cosine_similarity.load_cosine_similarities(cosine_similarities_path)
    documents = json.load(open(labeled_documents_path))
    print("results are loaded :)")

