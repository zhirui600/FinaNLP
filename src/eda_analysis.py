# -*- coding: utf-8 -*-
import os
import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import umap

from config import DATA_PATH, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_eda_and_interpretability():
    logger.info("=" * 60)
    logger.info("RUNNING EDA & INTERPRETABILITY")
    logger.info("=" * 60)
    
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        logger.error(f"Could not load data for EDA: {e}")
        return
        
    df["word_count"] = df["Sentence"].apply(lambda x: len(str(x).split()))
    
    # 2. EDA: Class Distribution
    plt.figure(figsize=(8, 5))
    order = ["positive", "neutral", "negative"]
    sns.countplot(data=df, x="Sentiment", order=order, palette="Set2")
    plt.title("Class Distribution in Dataset")
    plt.ylabel("Count")
    
    val_counts = df["Sentiment"].value_counts()
    for i, sentiment in enumerate(order):
        if sentiment in val_counts:
            v = val_counts[sentiment]
            plt.text(i, v + 20, str(v), ha="center")
            
    plt.savefig(os.path.join(results_dir, "eda_class_distribution.png"), dpi=150)
    plt.close()
    
    # 3. EDA: Sentence Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="word_count", hue="Sentiment", multiple="stack", bins=30, palette="Set2")
    plt.title("Sentence Length Distribution by Sentiment")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.savefig(os.path.join(results_dir, "eda_sentence_length.png"), dpi=150)
    plt.close()
    
    # 4. Word Clouds
    plt.figure(figsize=(15, 5))
    for i, sentiment in enumerate(["positive", "neutral", "negative"]):
        text = " ".join(df[df["Sentiment"] == sentiment]["Sentence"].astype(str))
        wc = WordCloud(width=400, height=300, background_color="white", max_words=100).generate(text)
        
        plt.subplot(1, 3, i+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.title(f"{sentiment.capitalize()} Sentiment")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "wordcloud_by_class.png"), dpi=150)
    plt.close()
    
    # 5. TF-IDF Feature Importance
    models_dir = os.path.join(OUTPUT_DIR, "models")
    try:
        with open(os.path.join(models_dir, "ml_best_model.pkl"), "rb") as f:
            ml_model = pickle.load(f)
            
        vectorizer = ml_model.named_steps["tfidf"]
        clf = ml_model.named_steps["clf"]
        
        feature_names = vectorizer.get_feature_names_out()
        
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            
            with open(os.path.join(models_dir, "label_mapping.pkl"), "rb") as f:
                mapping = pickle.load(f)
                idx2label = mapping["idx2label"]
                
            plt.figure(figsize=(15, 10))
            for i in range(coef.shape[0]):
                class_label = idx2label[i] if coef.shape[0] > 1 else "Positive (Binary)"
                class_coef = coef[i]
                
                top_positive_idx = np.argsort(class_coef)[-10:]
                top_negative_idx = np.argsort(class_coef)[:10]
                
                top_idx = np.concatenate([top_negative_idx, top_positive_idx])
                top_features = [feature_names[j] for j in top_idx]
                top_coefs = [class_coef[j] for j in top_idx]
                
                colors = ["red" if c < 0 else "green" for c in top_coefs]
                
                plt.subplot(coef.shape[0], 1, i+1)
                sns.barplot(x=top_coefs, y=top_features, palette=colors)
                plt.title(f"Top TF-IDF Features for Class: {class_label}")
                
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "tfidf_top_features.png"), dpi=150)
            plt.close()
            
    except Exception as e:
        logger.warning(f"Could not extract TF-IDF feature importance: {e}")
        
    # 6. UMAP visualization of TF-IDF vectors
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        X = tfidf.fit_transform(sample_df["Sentence"]).toarray()
        
        reducer = umap.UMAP(random_state=42, n_components=2)
        embedding = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=sample_df["Sentiment"], 
                        palette="Set1", s=30, alpha=0.7)
        plt.title("UMAP Projection of TF-IDF Vectors")
        plt.savefig(os.path.join(results_dir, "tsne_tfidf.png"), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate UMAP projection: {e}")
        
    logger.info("EDA & Interpretability complete.")

if __name__ == "__main__":
    run_eda_and_interpretability()
