from sklearn.decomposition import PCA
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os
import joblib
import logging
import boto3


def save_pca_model(pca_model, filename):
    joblib.dump(pca_model, filename)

def load_pca_model(filename):
    logging.info("PCA Model" + str(filename))
    return joblib.load(filename)

def generate_embeddings_pca(df, column_names, pca_models_dir, pca_models=False):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    embeddings_df = pd.DataFrame()  # Initialize an empty DataFrame for embeddings

    # Check if PCA models directory exists, if not, create it
    if not os.path.exists(pca_models_dir):
        os.makedirs(pca_models_dir)

    for column_name in column_names:
        sentences = df[column_name].tolist()
        embeddings = model.encode(sentences)

        pca_model_path = os.path.join(pca_models_dir, f"{column_name}_pca_model.pkl")

        if not pca_models:
            # Create and fit a new PCA model to explain 90% of variance
            pca = PCA(n_components=0.9, svd_solver='full')
            pca.fit(embeddings)
            reduced_embeddings = pca.transform(embeddings)

            col_names = [f"{column_name}_embedding_{i}" for i in range(reduced_embeddings.shape[-1])]
            embeddings_col_df = pd.DataFrame(reduced_embeddings, columns=col_names)

            embeddings_df[col_names] = embeddings_col_df

            # Save PCA model to a pickle file
            save_pca_model(pca, pca_model_path)
        else:
            # Load existing PCA model from pickle file
            pca = load_pca_model(pca_model_path)

            reduced_embeddings = pca.transform(embeddings)

            col_names = [f"{column_name}_embedding_{i}" for i in range(reduced_embeddings.shape[-1])]
            embeddings_col_df = pd.DataFrame(reduced_embeddings, columns=col_names)

            embeddings_df[col_names] = embeddings_col_df

    return embeddings_df
