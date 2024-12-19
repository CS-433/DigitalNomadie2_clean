
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

def load_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return tweets

def load_and_clean_test_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    cleaned_tweets = [re.sub(r"^\d+,\s*", "", tweet).strip() for tweet in tweets]
    return cleaned_tweets

def count_metric(tweet, vocab, count_words=True):
    if count_words:
        return len(tokenize_with_vocab(tweet, vocab))
    else:
        return len(tweet)  

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

def tokenize_with_vocab(tweet, vocab):
    return [word for word in tweet.split() if word in vocab]

def tweet_length_plotter(data_paths, clean_paths, vocab_path, count_words=True, normalize=True, plot_type='hist'):
    vocab = load_vocab(vocab_path)

    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    all_datasets = datasets + cleaned_datasets
    dataset_labels = ['Train Negative Tweets', 'Train Positive Tweets', 'Test Tweets (Cleaned)']
    
    dataset_lengths = [
        [count_metric(tweet, vocab, count_words) for tweet in dataset] 
        for dataset in all_datasets
    ]
    
    plt.figure(figsize=(12, 6))
    metric = 'Words' if count_words else 'Characters'

    if plot_type == 'hist':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            plt.hist(lengths, bins=50, alpha=0.5, label=label, color=color, density=normalize)
        plt.ylabel('Density' if normalize else 'Number of Tweets',fontsize=18)
    elif plot_type == 'kde':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            sns.kdeplot(lengths, fill=True, label=label, color=color)
        plt.ylabel('Density',fontsize=18)
    else:
        raise ValueError("Invalid plot_type. Use 'hist' or 'kde'.")
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(f'Number of {metric} per Tweet',fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def count_tweets_above_word_threshold(data_paths, clean_paths, word_threshold):
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    all_datasets = datasets + cleaned_datasets
    
    total_count = sum(
        1 for dataset in all_datasets for tweet in dataset 
        if len(tweet.split()) > word_threshold
    )
    
    return total_count

def ratio_tweets_above_word_threshold(data_paths, clean_paths, word_threshold):
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    all_datasets = datasets + cleaned_datasets
    
    all_tweets = [tweet for dataset in all_datasets for tweet in dataset]
    
    count_above_threshold = sum(1 for tweet in all_tweets if len(tweet.split()) > word_threshold)
    
    total_tweets = len(all_tweets)
    ratio = count_above_threshold / total_tweets if total_tweets > 0 else 0
    
    return ratio

from collections import Counter

def top_100_words_from_datasets(data_paths, clean_paths, vocab_path, min_word_length=1):
    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocabulary = set(line.strip() for line in file)  

    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    all_datasets = datasets + cleaned_datasets
    
    word_list = []
    for dataset in all_datasets:
        for tweet in dataset:
            words = tweet.split()
            valid_words = [
                word for word in words 
                if word in vocabulary and len(word) >= min_word_length
            ]
            word_list.extend(valid_words)
    
    word_counts = Counter(word_list)
    
    top_100_words = word_counts.most_common(100)
    
    return top_100_words


