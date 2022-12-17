import pandas as pd
import re
import numpy as np


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
            new_values.append("<INT_VAR>" if value.isdigit() else "<STR_VAR>")
            #new_values.append(value)
            if len(delimiters)>index :
                new_values.append(delimiters[index])
        
        new_token_list = [*new_token_list, param,"=", *new_values]
    return new_token_list

data_df["URL"]= data_df["URL"].apply( lambda x : tokenize_url(x) )
data_df["content"] = data_df["content"].apply(lambda x : tokenize_post_body(x) if x is not None else x)

sequence_df = data_df.apply(lambda x : ((x[0] + x[1]) if x[1] is not None else x[0]) + ["<pad>"]*(309-(len(x[0]) + len(x[1]) if x[1] is not None else len(x[0]))), axis = 1)

def generate_freq_dist(sequence_df, token_dict) :
    for token_list in sequence_df :
        if token_list is not None :
            for token in token_list : 
                token_dict[token] = (token_dict[token] if token in token_dict else 0 )+ 1
            
token_dict = {}
generate_freq_dist(sequence_df ,token_dict)

token_dict = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse = True)}

vocab = {}
count = 0 
for item in token_dict:
    vocab[item] = count
    count = count + 1

inverse_vocab = {index: token for token, index in vocab.items()}

metadata_df = pd.read_csv('metadata_WithVariables.tsv', sep = '\t', header = None)
vectors_df = pd.read_csv('vectors_WithVariables.tsv', sep = '\t', header = None)

def generate_embedding(index):
    vector = sequence_df.iloc[index]
    embedded_vector = []
    for word in vector :  
        embedded_vector.append(vectors_df.iloc[vocab[word]])
    return np.array(embedded_vector)    
    
    
#print(generate_embedding(13))

