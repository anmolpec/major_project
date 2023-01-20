import pandas as pd
from urllib.parse import unquote
import numpy as np
import re


df = pd.read_csv('../data/csic_database.csv')
data_df = pd.DataFrame()
data_df["classification"] = df['classification']
merge_url_content = lambda x : x['URL'].split()[0] + (('&' if "?" in x['URL'] else "?" + x['content']) if not pd.isna(x['content']) else '')
data_df["URL"]= df.apply( merge_url_content , axis = 1)


#V3

#some s+cript not being caught 49579



def multisplit(regex, string):
    return re.split(regex, string)

def get_delimiter(regex, string):
    return re.findall(regex, string)

def tokenize_url(x):
    x = unquote(x, encoding='cp1252') #45589
    x = unquote(x, encoding='cp1252') #double encoded url string (should probably have this in a loop)
    delimiter_string = "[?&=@\.%+<>()/~;'-*!#,:]|\r|\n" #could add _ but used in pswd like line 45584   "" are a problem 45728
    delimiters = get_delimiter(delimiter_string, x)
    tokens = multisplit(delimiter_string , x)
    new_url_tokens = []
    for index,token in enumerate(tokens):
        new_url_tokens.append(token.lower())
        if index < len(delimiters) :
            new_url_tokens.append(delimiters[index])
    new_url_tokens = [token for token in new_url_tokens if len(token)>0]  
    new_url_tokens = new_url_tokens + ['<pad>']*(309-len(new_url_tokens))
    return new_url_tokens

data_df["URL"]= data_df["URL"].apply( lambda x : tokenize_url(x) )


def generate_freq_dist(sequence_df, token_dict) :
    for token_list in sequence_df :
        if token_list is not None :
            for token in token_list : 
                token_dict[token] = (token_dict[token] if token in token_dict else 0 )+ 1
            
token_dict = {}
generate_freq_dist(data_df['URL'] ,token_dict)
token_dict = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse = True)}

vocab = {}
count = 0 
for item in token_dict:
    vocab[item] = count
    count = count + 1
    
inverse_vocab = {index: token for token, index in vocab.items()}

temp = list(data_df['URL'])
sequences = []
for temp_seq in temp:
    seq = []
    for token in temp_seq : 
        seq.append(vocab[token])
    sequences.append(seq)


vectors_df = pd.read_csv('vectors_v2.0.tsv', sep = '\t', header = None)

def generate_embedding(index):
    word_vector = data_df['URL'].iloc[index]
    embedded_vector = []
    for word in word_vector :  
        embedded_vector.append(vectors_df.iloc[vocab[word]])
    return np.array(embedded_vector)   

