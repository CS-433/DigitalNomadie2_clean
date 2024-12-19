# Harnessing Transformer Models for Superior Performance in Binary Classification for Twitter

This project aims to classify tweets, specifically whether they contain a positive ":)" or negative smiley ":(", using machine learning and natural language processing (NLP) techniques. By leveraging pre-trained word embeddings and advanced language models, the goal is to classify tweets based on their sentiment while addressing the unique challenges posed by the informal and noisy nature of Twitter data.

## Getting Started

Before running any scripts, ensure all dependencies are installed by running `requirements.txt`:
```bash
pip install -r requirements.txt
```


## Workflow Overview

The project is organized into four main stages:

1. **Embedding Generation and Preprocessing**:
   - The `save_embeddings.py` script generates vector representations for tweets using models like DistilBERT and `all-MiniLM-L6-v2`. It transforms tweets into fixed-length feature vectors and saves them for model training.
   - For GloVe embeddings, the process differs from other models. The script `create_features_glove.py` is used to generate GloVe-based embeddings, leveraging resources provided in the `Provided` folder, (the instruction ar in the `README.md` provided in this folder).

2. **Model Training and Evaluation**:
   - The `classifier_optimization.py` script trains and evaluates classifying models (e.g., logistic regression, multi-layer perceptron) on the generated embeddings to classify tweets as positive or negative. It also includes hyperparameter tuning for performance optimization.

3. **Fine-Tuning and Advanced Evaluation**:
   - The `fine_tuning.py` script performs the fine-tuning of an array of LLMs, including `bertweet-base`, a pre-trained language model specialized for Twitter sentiment analysis. Fine-tuning enhances its performance for classifying sentiment in Twitter-specific text.

4. **Generate Submission File**:
   - The final script `run.py` generates predictions on the test dataset using the fine-tuned model and creates a `.csv` submission file compatible with evaluation platforms.

---

### Step 1: Generate Tweet Embeddings

- The `save_embeddings.py` script processes raw tweet data, extracts features using pre-trained embedding models, and saves the results for training and evaluation. The output is a feature matrix enriched with the tweet length (word count) as an additional feature.
- For GloVe embeddings, the `create_features_glove.py` script must be used. This script utilizes the GloVe model provided in the `Provided` folder to generate embeddings tailored to the dataset.

#### Model Selection:
You can choose from pre-trained models such as:
- `sentence-transformers/all-MiniLM-L6-v2`
- `vinai/bertweet-base`
- `distilbert-base-uncased`
- `cardiffnlp/twitter-roberta-base-sentiment`

Sentence-transformers models use the `SentenceTransformer` library, while other models rely on Hugging Face’s `AutoTokenizer` and `AutoModel`.

#### Output:

1. A .npy file containing the feature matrix (embeddings with word count):
   features_all_<model_name>.npy

2. A .npy file containing the labels:
   labels_all_<model_name>.npy

Example Results:
Using the vinai/bertweet-base model, the script outputs:
- features_all_bertweet-base.npy: A matrix of tweet embeddings with additional word count feature.
- labels_all_bertweet-base.npy: Labels corresponding to the smiley present in the tweets.

This step generates the necessary input for training classifiers, allowing you to experiment with various models and configurations in the subsequent steps.


### Step 2: Train and Evaluate the Model

The `classifier_optimization.py` script trains and evaluates different classifiers using embeddings generated in Step 1. You can select both the embedding model and classifier to use.

#### Model and Classifier Options:
- The embedding models:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `vinai/bertweet-base` (best results)
  - `distilbert-base-uncased`
  - `cardiffnlp/twitter-roberta-base-sentiment`
- The classifiers:
  - Logistic Regression: Optimizes regularization parameter.
  - Linear Regression: Directly fits the model and thresholds predictions.
  - MLP Classifier: Optimizes hidden layer size and learning rate.

#### How to Run:
1. Set the embedding model and classifier:
Example : 
model_name = "vinai/bertweet-base" 
classifier = "MLPClassifier"
#### Output:

- Best hyperparameters for the selected classifier.
- Test set performance metrics (F1 score and accuracy).

#### Example Results

For example, running the script with `vinai/bertweet-base` embeddings and the `MLPClassifier` might yield:
Optimal parameters and results for MLP classifier:
Best Parameters: {'hidden_dim': 100, 'learning_rate': 0.001}
Test set F1 Score: 0.87
Test set accuracy: 0.89

This modular design allows you to experiment with different combinations of embedding models and classifiers to find the best setup for your sentiment analysis task.

### Step 3: Fine-Tune the Pre-trained Model

The `fine_tuning.py` script fine-tunes pre-trained language models (e.g., `vinai/bertweet-base`, `twitter-roberta-base-sentiment`) for tweet sentiment classification. It uses positive (`train_pos_full.txt`) and negative (`train_neg_full.txt`) tweets, assigns labels (`1` for positive, `-1` for negative), and splits the data into training and evaluation sets.

The script adapts the model to Twitter-specific data through transfer learning, leveraging Hugging Face’s `Trainer`. After training, the fine-tuned model and tokenizer are saved in a directory, and evaluation metrics are displayed.

#### Output:

The fine-tuned model and tokenizer are saved for future use:
./fine_tuned_<model_name>

Evaluation results are printed to the console after training:
{'eval_loss': <value>, 'eval_accuracy': <value>}

This script demonstrates the effectiveness of transfer learning by fine-tuning a pre-trained model on a domain-specific task. It provides a strong foundation for sentiment analysis and other classification tasks on Twitter data.

### Step 4: Generate Submission File

The `generate_submission.py` script generates sentiment predictions for tweets in the test dataset using a fine-tuned model and creates a submission file in CSV format.

#### How the Script Works:

1. **Load the Fine-Tuned Model and Tokenizer**:
   - The model and tokenizer are loaded from the directory specified in `model_name` (e.g., `fine_tuned_bertweet-base`).

2. **Predict Sentiment**:
   - Tweets from `test_data.txt` are tokenized and processed by the model.
   - Predictions are based on the computed probabilities:
     - `1` for positive sentiment.
     - `-1` for negative sentiment.

3. **Create Submission File**:
   - Each tweet is assigned a unique ID starting from 1.
   - The `helpers.py` module is used to call `create_csv_submission`, which saves the predictions and IDs in a CSV file named `submission_<model_name>.csv`.

#### Output:
A `.csv` file is saved in the `data/` directory (e.g., `data/submission_bertweet-base.csv`) containing:
- **Id**: The unique ID of each tweet.
- **Prediction**: The predicted sentiment (`1` for positive, `-1` for negative`).

The `helpers.py` file is required to properly format the submission file.


## Additionnal files 


### Data Folder

The `data` folder contains the `twitter-dataset`, which includes:

- **`train_pos.txt`**: Contains 10% tweets labeled as positive.
- **`train_neg.txt`**: Contains 10% tweets labeled as negative.
- **`train_pos_full.txt`**: Contains tweets labeled as positive.
- **`train_neg_full.txt`**: Contains tweets labeled as negative.

These files are used for embedding generation and for the fine tuning. The two full ones aren't in the Git repository because they are too large but available in the AIcrowd ressources.

- **`test_data.txt`**: Contains tweets for which sentiment predictions need to be generated.

This file is used to create submission.

### Data Exploration

The `data_exploration` folder contains:

1. **`data_exploration.ipynb`**:
   - A Jupyter Notebook for analyzing the dataset (e.g., class balance, tweet length).
   - Calls functions from `data_exploration_helpers.py`.

2. **`data_exploration_helpers.py`**:
   - A module with helper functions used in the notebook.

Use the notebook to interactively explore and understand the dataset.

### `vocab_cut.txt`
The `vocab_cut.txt` file is specifically used during GloVe embedding generation and for data exploration tasks.