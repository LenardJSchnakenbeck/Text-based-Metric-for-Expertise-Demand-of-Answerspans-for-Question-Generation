from preprocessing import candidate_phrases
from similarity import cosine_similarity

source_documents_path = 'preprocessing/wikipedia_texts.json'
labeled_documents_path = 'preprocessing/labeled_documents.json'
embeddings_path = 'similarity/embeddings.pickle'
cosine_similarities_path = 'similarity/cosine_similarities.pickle'

#chunk candidates by pos-tag
candidate_phrases.write_json_labeled_documents(source_documents_path, labeled_documents_path)

#calculate similarities (candidates-candidates & candidates-documents)
cosine_similarity.write_pickle_encoded_documents(embeddings_path, labeled_documents_path)
cosine_similarity.calculate_and_write_pickle_cossim(labeled_documents_path, embeddings_path, cosine_similarities_path)

cossim_documents = cosine_similarity.load_cosine_similarities(cosine_similarities_path)

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
