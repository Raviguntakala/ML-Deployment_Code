## importing necessary packages
import nltk
import pickle
import warnings
import json
import os
import numpy as np
import pandas as pd
from typing import List,Dict,Any
import modules.preprocess as preprocess
from modules.embeddings import generate_embeddings_pca
from flask import Flask, request, jsonify
import joblib
import logging
import boto3
import time


# nltk.download('omw-1.4')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category=DeprecationWarning)

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
logging.info("Model Path" + str(model_path))

def hcid_classifier(request_body: dict,
                   text_columns: List[str] = ['Summary', 'Description'],
                   model_dir: str = model_path) -> Dict[str, Any] :

    """
    This function classifies a dataframe based on the provided test columns using a pre-trained machine learning model.

    Parameters:
    df (pd.DataFrame): The dataframe to be classified.
    text_columns (List[str]): A list of column names in the dataframe to be used for classification.
    preprocess_params (Dict[str, Any]): A dictionary of parameters for preprocessing the data.
    pca_model_dir (str): The directory path of the pre-trained PCA model.
    ml_model (Any): The pre-trained machine learning model.

    Returns:
    A dictionary with the following keys:
    'successIndicator' (bool): A boolean indicating whether the classification was successful.
    'Label' (str): The predicted label for the dataframe.
    'Confidence Score' (float): The confidence score of the prediction.
    """


    # Convert JSON reponse into Pandas Dataframe
    df = pd.DataFrame([request_body])

    # Reading config file
    with open("modules/config.json","r") as file:
        config = json.load(file)
        preprocess_param = config['preprocess_param']
        
    # check if ticket belongs WGS group or not
    if df['Platform'].isin(['WGS Local/WGS Host','WGS National']).any():

        #preprocess text columns data
        for col in text_columns:
            df[col] = df[col].apply(lambda x:preprocess.preprocess(x, training=True, params=preprocess_param))

        #create embeddings
        embeddings = generate_embeddings_pca(df=df,column_names =text_columns,pca_models_dir=model_dir,pca_models=True)

        #load HCID model
        model = joblib.load(os.path.join(model_dir, 'HCID.pkl'))
        logging.info("Model Config" + str(model))



        # predict the category
        predicted_labels = model.predict(embeddings)

        # predict the probablity
        predicted_prob = model.predict_proba(embeddings)
        probability = round(predicted_prob[0][np.argmax(predicted_prob)])

        # generate response
        response = {"successIndicator" : True,
                    "label" : config[predicted_labels[0]],
                    "confidence" : probability }

    else:
        response = {"successIndicator" : True,
                    "label" : "Others",
                    "confidence" : 'N/A'}

    return json.dumps(response,indent = 4, default=int)


app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    """Apply security headers to the response."""
    # Apply HSTS
    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'
    return response

@app.route('/ping', methods=['GET'])
def health():
    # Check if the classifier was loaded correctly
    try:
        #regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return jsonify(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Endpoint to classify the input data using the HCID classification model.
    """
    try:

        request_body = request.get_json()

        # Classify the input data
        response = hcid_classifier(request_body)

        return response, 200

    except Exception as e:
        return jsonify({"Status":500,"successIndicator" : False,"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
