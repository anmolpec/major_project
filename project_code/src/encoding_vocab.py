from glob import glob
import numpy as np
import pandas as pd
import re

embedding_dim = 300
df = pd.read_csv('../data/csic_database.csv')
data_df = df.loc[0:,["URL","content","classification"]]
data_df["URL"]= data_df["URL"].apply( lambda x : x.split()[0] )
data_df["content"]= data_df["content"].apply( lambda x : None if pd.isna(x) else x )



def multisplit(regex, string):
    return re.split(regex, string)

def get_delimiter(regex, string):
    return re.findall(regex, string)

def tokenize_url(x):
    if "?" in x :
        url, query = x.split("?")
    else :
        url = x
        query = None
        
    url_tokens = multisplit("[/:\.]", url)
    url_delimiters = get_delimiter("[/:\.]", url)
    new_url_tokens = []
    for index,value in enumerate(url_tokens) :
            new_url_tokens.append(value)
            if len(url_delimiters)>index :
                new_url_tokens.append(url_delimiters[index])
                
    new_url_tokens = list(filter(lambda x : len(x)>0 , new_url_tokens))
    
    query_tokens = tokenize_post_body(query) if query is not None else []
    
    token_list = [*new_url_tokens, "?" , *query_tokens] if query is not None else [*new_url_tokens]
    
    return token_list

def tokenize_post_body(x):
    token_list = x.split("&")
    new_token_list = []
    for index, token in enumerate(token_list) :
        delimiters = get_delimiter("[@\.%+]", token)
        token = multisplit("[=@\.%+]" , token)
        param, *values = token
        new_values = []
        for index,value in enumerate(values) :
            #new_values.append("<INT_VAR>" if value.isdigit() else "<STR_VAR>")
            new_values.append(value)
            if len(delimiters)>index :
                new_values.append(delimiters[index])
        
        new_token_list = [*new_token_list, param,"=", *new_values]
    return new_token_list

data_df["URL"]= data_df["URL"].apply( lambda x : tokenize_url(x) )
data_df["content"] = data_df["content"].apply(lambda x : tokenize_post_body(x) if x is not None else x)


def generate_dict(token_df, token_set) :
    for token_list in token_df :
        if token_list is not None :
            for token in token_list : 
                token_set.add(token.lower())
            
token_set = set(())
generate_dict(data_df.loc[0:,"URL"] ,token_set)
generate_dict(data_df.loc[0:,"content"] ,token_set)


def generate_indexed_vocab(token_set):
    vocab = {}
    count = 0
    for token in token_set:
        vocab[token.lower()] = count
        count = count + 1
    return vocab

vocab = generate_indexed_vocab(token_set)


def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index)
      
    embedding_matrix_vocab = np.random.rand(vocab_size, embedding_dim)#np.zeros((vocab_size, embedding_dim))
  
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
                
  
    return embedding_matrix_vocab


embedding_matrix_vocab = embedding_for_vocab('../data/glove.42B.300d.txt', vocab ,embedding_dim)






