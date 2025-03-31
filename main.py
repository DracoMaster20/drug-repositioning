from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

main = Flask(__name__)

# Load pickled objects
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/node_embedding_dict.pkl", "rb") as f:
    node_embedding_dict = pickle.load(f)

with open("models/drugs_df.pkl", "rb") as f:
    drugs_df = pickle.load(f)

with open("models/disease_targets_df.pkl", "rb") as f:
    disease_targets_df = pickle.load(f)

# Build a mapping from disease names to disease IDs
# Assumes 'DiseaseName' and 'DiseaseID' columns exist in disease_targets_df.
disease_name_to_id = dict(zip(disease_targets_df["DiseaseName"], disease_targets_df["DiseaseID"]))


def predict_drugs_and_proteins_for_disease(disease_name):
    """
    Given a disease name, this function retrieves its corresponding ID,
    computes the prediction scores for all drugs, and returns the top 10 drugs
    with their IDs, names, associated proteins (drug targets), and prediction score.
    """
    if disease_name not in disease_name_to_id:
        return None, "Disease not found in the dataset."
    
    disease_id = disease_name_to_id[disease_name]
    
    # Ensure that the disease_id has an embedding in the graph.
    if disease_id not in node_embedding_dict:
        return None, "Disease ID not found in the graph embeddings."
    
    disease_embedding = node_embedding_dict[disease_id]
    drug_scores = {}
    drug_info = {}  # Dictionary mapping drug ID to (drug name, proteins affected)
    
    # Iterate over each drug in the drugs dataframe.
    for _, row in drugs_df.iterrows():
        drug_id = row["DrugID"]
        # Use DrugName if available; otherwise, default to drug_id
        drug_name = row["DrugName"] if "DrugName" in row and pd.notna(row["DrugName"]) else drug_id
        
        # Check if the drug has an embedding (convert drug_id to string)
        if str(drug_id) in node_embedding_dict:
            drug_embedding = node_embedding_dict[str(drug_id)]
            feature_vector = np.concatenate([disease_embedding, drug_embedding]).reshape(1, -1)
            # Get the prediction probability for a positive link (association)
            score = rf_model.predict_proba(feature_vector)[0, 1]
            drug_scores[drug_id] = score
            
            # Process the proteins affected (DrugTarget); assume comma-separated if multiple.
            proteins = row["DrugTarget"]
            if pd.isna(proteins):
                proteins = []
            else:
                proteins = [prot.strip() for prot in str(proteins).split(',')]
            drug_info[drug_id] = (drug_name, proteins)
    
    # Sort the drugs based on prediction scores (highest first) and pick the top 10.
    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    top_drugs = []
    for drug_id, score in sorted_drugs[:10]:
        name, proteins = drug_info.get(drug_id, (drug_id, []))
        top_drugs.append({
            "DrugID": drug_id,
            "DrugName": name,
            "Proteins": proteins,
            "Score": score
        })
    
    return top_drugs, None


@main.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    if request.method == 'POST':
        disease_name = request.form.get('disease_name')
        results, error = predict_drugs_and_proteins_for_disease(disease_name)
    return render_template('index.html', results=results, error=error, diseases=sorted(disease_name_to_id.keys()))


if __name__ == '__main__':
    main.run(debug=True)
