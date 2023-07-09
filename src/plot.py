import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():


    # Read the JSON file
    with open('data.json', 'r') as f:
        data = json.load(f)

    # Extract file paths, labels, and embeddings from the JSON data
    file_paths = []
    labels = []
    embeddings = []

    for file_path, value in data.items():
        file_paths.append(file_path)
        labels.append(value['label'])
        embeddings.append(value['embedding'][0])

    labels = np.array(labels)
    embeddings = np.array(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create a scatter plot of the embeddings
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels)
    plt.title('Image Embeddings in Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Save the plot as a .png image
    plt.savefig('scatter_plot.png')

if __name__ == "__main__":
    main()