import nltk

def candidate_chunking(documents, candidates):
    text = "Hello my name is your mom"
    tokens = nltk.word_tokenize(text)

    #tokens = documents["tokenized_text"]
    max_candidate_length = max([len(nltk.word_tokenize(can)) for can in candidates]) # +3,bc .split() != tokenization
    chunks = []
    candidate_index = 0
    #Iterate through all tokens
    i = 0
    while i < len(tokens):
        if candidate_index < len(candidates):
            candidate = candidates[candidate_index]
            if tokens[i] in candidate:
                c = []
                for j in range(len(nltk.word_tokenize(candidate))+1):
                    if nltk.word_tokenize(candidate) == tokens[i: min(i+j, len(tokens))]:
                        c = tokens[i:min(i+j, len(tokens))]
                if c:
                    chunks += [[c, True]]
                    candidate_index += 1
                    i += len(c)-1
            else:
                chunks += [[tokens[i], False]]
        else:
            chunks += [[tokens[i], False]]
        i += 1
    return chunks

print(candidate_chunking(None, ["my name is", "your mom"]))