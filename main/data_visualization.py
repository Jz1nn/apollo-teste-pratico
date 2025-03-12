import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

def visualize_embeddings(df):
    embeddings = np.vstack(df['embedding'].values)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    df['embedding_2d_x'] = embeddings_2d[:, 0]
    df['embedding_2d_y'] = embeddings_2d[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='embedding_2d_x', y='embedding_2d_y', hue='syndrome_id', data=df, palette='tab10')
    plt.title('t-SNE visualization of embeddings')

    # Save the plot
    results_dir = 'results'
    plt.savefig(os.path.join(results_dir, 'visualization_of_embeddings.png'))
    
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    visualize_embeddings(df)