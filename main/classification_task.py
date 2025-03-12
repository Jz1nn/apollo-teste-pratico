import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def classify_embeddings(df):
    X = np.vstack(df['embedding'].values)
    y = df['syndrome_id'].values
    
    param_grid = {'n_neighbors': list(range(1, 16))}
    
    knn_cosine = KNeighborsClassifier(metric='cosine')
    grid_search_cosine = GridSearchCV(knn_cosine, param_grid, cv=10, scoring='accuracy')
    grid_search_cosine.fit(X, y)
    best_k_cosine = grid_search_cosine.best_params_['n_neighbors']
    print(f'Best k for cosine distance: {best_k_cosine}')
    
    knn_euclidean = KNeighborsClassifier(metric='euclidean')
    grid_search_euclidean = GridSearchCV(knn_euclidean, param_grid, cv=10, scoring='accuracy')
    grid_search_euclidean.fit(X, y)
    best_k_euclidean = grid_search_euclidean.best_params_['n_neighbors']
    print(f'Best k for euclidean distance: {best_k_euclidean}')
    
    knn_cosine_best = KNeighborsClassifier(n_neighbors=best_k_cosine, metric='cosine')
    knn_euclidean_best = KNeighborsClassifier(n_neighbors=best_k_euclidean, metric='euclidean')
    
    return knn_cosine_best, knn_euclidean_best

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    knn_cosine_best, knn_euclidean_best = classify_embeddings(df)