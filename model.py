#Importing the necessary libraries
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import random

# Loading the datasets
drugs_df = pd.read_csv("/content/drive/Shareddrives/Drug_repositioning/drugsInfo.csv")
disease_targets_df = pd.read_csv("/content/drive/Shareddrives/Drug_repositioning/Disease_with_DrugTarget.csv")
mapping_df = pd.read_csv("/content/drive/Shareddrives/Drug_repositioning/mapping.csv")

# Creating the graph instance
G = nx.DiGraph()

# Adding Drugs and Diseases as nodes
G.add_nodes_from(drugs_df["DrugID"], node_type="drug")
G.add_nodes_from(disease_targets_df["DiseaseID"], node_type="disease")

# Adding Targets as nodes
unique_targets = set(drugs_df["DrugTarget"].dropna().unique())
G.add_nodes_from(unique_targets, node_type="target")

# Adding Drug-Target edges
drug_target_edges = list(zip(drugs_df["DrugID"], drugs_df["DrugTarget"]))
G.add_edges_from(drug_target_edges, relation="interacts_with")

# Adding Target-Disease edges
target_disease_edges = list(zip(disease_targets_df["DrugTargets"], disease_targets_df["DiseaseID"]))
G.add_edges_from(target_disease_edges, relation="associated_with")

# Adding Drug-Disease known interactions
drug_disease_edges = list(zip(mapping_df["DrugID"], mapping_df["DiseaseID"]))
G.add_edges_from(drug_disease_edges, relation="treats")

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Generating the Node Embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
node_embedding_dict = {str(node): model.wv[str(node)] for node in G.nodes()}

# Creating positive and negative samples
positive_edges = drug_disease_edges
negative_edges = set()
drug_nodes = list(drugs_df["DrugID"].unique())
disease_nodes = list(disease_targets_df["DiseaseID"].unique())

while len(negative_edges) < len(positive_edges):
    d = random.choice(drug_nodes)
    dis = random.choice(disease_nodes)
    if (d, dis) not in G.edges():
        negative_edges.add((d, dis))
negative_edges = list(negative_edges)

# Performing Train-Test Split for edges
edges = positive_edges + negative_edges
labels = [1] * len(positive_edges) + [0] * len(negative_edges)
edges_train, edges_test, labels_train, labels_test = train_test_split(edges, labels, test_size=0.2, random_state=42)

# Converting edges to feature vectors
def edge_to_feature(edge_list, embedding_dict):
    return [np.concatenate([embedding_dict.get(str(edge[0]), np.zeros(64)), embedding_dict.get(str(edge[1]), np.zeros(64))]) for edge in edge_list]

#Creating train and test inputs
X_train = edge_to_feature(edges_train, node_embedding_dict)
X_test = edge_to_feature(edges_test, node_embedding_dict)




# Training the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, labels_train)

# Evaluating the model
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(labels_test, predictions)
roc_auc = roc_auc_score(labels_test, predictions)
print(f"Accuracy: {accuracy:.4f}, AUC-ROC: {roc_auc:.4f}")


import joblib

# Save Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# Save Node2Vec model
model.wv.save_word2vec_format("node2vec_embeddings.txt")

print("Model and Embeddings Saved Successfully!")