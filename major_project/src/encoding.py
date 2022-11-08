import numpy as np
from encoding_vocab import data_df, vocab, embedding_dim, embedding_matrix_vocab

def get_max_url_dim(data_df):
    max_dim = data_df.apply(lambda x : len(x[0]) + (len(x[1]) if x[1] is not None else 0), axis = 1).max()
    return max_dim

def generate_embedding(index):
    vector_len = get_max_url_dim(data_df)
    vector_depth = embedding_dim
    embedded_vector = np.zeros((vector_len, vector_depth))
    curr_index = 0
    url = data_df.iloc[index,0]
    body = data_df.iloc[index,1]
    curr_index = 0
    for token in url :
        token = token.lower()
        token_vector = embedding_matrix_vocab[vocab[token]]
        embedded_vector[curr_index] = token_vector
        curr_index = curr_index + 1
        
    if body is not None : 
        for token in body :
            token = token.lower()
            token_vector = embedding_matrix_vocab[vocab[token]]
            embedded_vector[curr_index] = token_vector
            curr_index = curr_index + 1
        
    return embedded_vector




