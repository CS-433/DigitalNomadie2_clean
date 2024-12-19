import numpy as np
import pandas as pd

with open("vocab_cut.txt", "r") as f:
    words = [line.strip() for line in f]

print(f"Number of words in vocab.txt: {len(words)}")

embedding_matrix = np.load("embeddings.npy")

glove_embeddings = {words[i]: embedding_matrix[i] for i in range(len(words))}

data_path = "data/twitter-datasets/"
train_neg_path = f"{data_path}train_neg.txt"
train_pos_path = f"{data_path}train_pos.txt"
test_path = f"{data_path}test_data.txt"

with open(train_neg_path, "r") as f:
    neg_tweets = [(line.strip(), -1) for line in f]

with open(train_pos_path, "r") as f:
    pos_tweets = [(line.strip(), 1) for line in f]

tweets_with_labels = neg_tweets + pos_tweets

import random
random.shuffle(tweets_with_labels)

df = pd.DataFrame(tweets_with_labels, columns=["tweet", "label"])

def get_average_embedding(tweet, glove_embeddings, embedding_dim=20):
    words = tweet.split()  
    word_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]

    if not word_vectors:
        return np.zeros(embedding_dim)
    
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector

def get_combined_feature(tweet, glove_embeddings, embedding_dim=20):
    avg_embedding = get_average_embedding(tweet, glove_embeddings, embedding_dim)
    
    tweet_length = len(tweet.split())
    
    combined_feature = np.append(avg_embedding, tweet_length)
    
    return combined_feature


embedding_dim = 20   
df["feature"] = df["tweet"].apply(lambda tweet: get_combined_feature(tweet, glove_embeddings, embedding_dim))


feature_matrix = np.vstack(df["feature"].values)
labels = df["label"].values

model_id = "glove"
filename = "features_all_{}.npy".format(model_id)
filename_labels = "labels_all_{}.npy".format(model_id)

np.save(filename, feature_matrix)
np.save(filename_labels, labels)