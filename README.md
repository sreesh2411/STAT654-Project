# Suicidal Text Detection in Social Media Posts

## Project Overview
This project focuses on identifying suicidal ideation in social media posts using Natural Language Processing (NLP) and deep learning techniques. Early diagnosis of suicide intent among adolescents is critical for intervention and therapy. We developed and evaluated multiple machine learning and transformer-based models to classify Reddit posts as either suicidal or non-suicidal. Additionally, a prototype Large Language Model (LLM) mental health chatbot was developed to provide therapeutic guidance.

## Dataset
The dataset was collected from Kaggle and contains posts from two subreddits: `r/SuicideWatch` and `r/depression`.
- **Total Posts:** 232,074 rows.
- **Original Distribution:** 50% labeled 'suicide' (from SuicideWatch) and 50% labeled 'non-suicide' (from teenagers and depression subreddits).
- **Cleaned Distribution:** After removing long posts, empty rows, and redundant words, the distribution shifted to approximately a 40:60 ratio of suicidal to non-suicidal texts.

## Methodology & Architecture
1. **Data Preprocessing:** Text preparation included spell checking with SymSpell, removing stop words, fixing word lengthening, lemmatization, and removing URLs, symbols, and digits.
2. **Word Embeddings:** A custom Word2Vec model was trained from scratch using the Continuous Bag-of-Words (CBOW) approach to capture semantic relationships specific to the dataset.
3. **Modeling:** Five different models were implemented to predict the binary variable:
   - Logistic Regression
   - Convolutional Neural Network (CNN) 
   - Long Short-Term Memory Neural Network (LSTM)
   - BERT (Bidirectional Encoder Representations from Transformers)
   - ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)

## Results
The transformer-based models significantly outperformed the traditional and standard deep learning models. 
- **Logistic Regression:** 90.54% Accuracy
- **CNN:** 92.24% Accuracy
- **LSTM:** 92.34% Accuracy
- **BERT:** 97.52% Accuracy
- **ELECTRA:** 97.61% Accuracy (Best Performing Model)

## Repository Structure
- `EDA.ipynb` & `preprocessing_cleaning.ipynb`: Notebooks for exploratory data analysis, data cleaning, and preprocessing text data.
- `Word2Vec-S.ipynb`: Training the custom Word2Vec embedding model.
- `Logit-Final.ipynb`: Baseline Logistic Regression implementation.
- `CNN-Final.ipynb`: Convolutional Neural Network implementation.
- `LSTM-Final.ipynb`: LSTM Neural Network implementation.
- `BERT.ipynb`: Fine-tuning the pre-trained BERT model.
- `Complete_Electra_transformer_huggingface_model (1).ipynb`: Fine-tuning the ELECTRA model.
- `finetuned_qlora_llama2_7b_sharded_final.ipynb`: Fine-tuning Meta's LLAMA 2 model using QLORA for the mental health chatbot.
- `STAT654_Project_Report.pdf`: Comprehensive report detailing the project's background, methodology, and results.
- `embedding_word2vec.txt` & `vocab.txt`: Exported custom Word2Vec embeddings and vocabulary.
- `mental_health_chatbot-.gif` & `resized_mental_health_chatbot.gif`: Visual demonstrations of the mental health chatbot prototype.

## Contributors
- Udbhav Srivastava
- Gunjan Joshi
- K. Sreesh Reddy
