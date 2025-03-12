import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize
import os

def evaluate_metrics(df, knn_cosine_best, knn_euclidean_best):
    X = np.vstack(df['embedding'].values)
    y = df['syndrome_id'].values
    
    knn_cosine_best.fit(X, y)
    knn_euclidean_best.fit(X, y)
    
    y_pred_cosine = knn_cosine_best.predict(X)
    y_pred_euclidean = knn_euclidean_best.predict(X)
    
    auc_cosine = roc_auc_score(y, knn_cosine_best.predict_proba(X), multi_class='ovr')
    auc_euclidean = roc_auc_score(y, knn_euclidean_best.predict_proba(X), multi_class='ovr')
    f1_cosine = f1_score(y, y_pred_cosine, average='weighted')
    f1_euclidean = f1_score(y, y_pred_euclidean, average='weighted')
    top_k_cosine = top_k_accuracy_score(y, knn_cosine_best.predict_proba(X), k=knn_cosine_best.n_neighbors)
    top_k_euclidean = top_k_accuracy_score(y, knn_euclidean_best.predict_proba(X), k=knn_euclidean_best.n_neighbors)
    
    print(f'Cosine distance AUC: {auc_cosine}, F1-score: {f1_cosine}, Top-{knn_cosine_best.n_neighbors} accuracy: {top_k_cosine}')
    print(f'Euclidean distance AUC: {auc_euclidean}, F1-score: {f1_euclidean}, Top-{knn_euclidean_best.n_neighbors} accuracy: {top_k_euclidean}')
    
    y_bin = label_binarize(y, classes=np.unique(y))
    
    plt.figure(figsize=(10, 8))
    for i in range(y_bin.shape[1]):
        fpr_cosine, tpr_cosine, _ = roc_curve(y_bin[:, i], knn_cosine_best.predict_proba(X)[:, i])
        fpr_euclidean, tpr_euclidean, _ = roc_curve(y_bin[:, i], knn_euclidean_best.predict_proba(X)[:, i])
        
        plt.plot(fpr_cosine, tpr_cosine, lw=2, label=f'Cosine distance ROC curve (class {i})')
        plt.plot(fpr_euclidean, tpr_euclidean, lw=2, linestyle='--', label=f'Euclidean distance ROC curve (class {i})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curves for Cosine and Euclidean Distance Metrics')
    plt.legend(loc="lower right")
    
    # Save the plot
    results_dir = 'results'
    plt.savefig(os.path.join(results_dir, 'roc_curve_comparison.png'))
    
    plt.show()

    # Generate summary table
    summary_table = pd.DataFrame({
        'Metric': ['AUC', 'F1-score', 'Top-k Accuracy'],
        'Cosine Distance': [auc_cosine, f1_cosine, top_k_cosine],
        'Euclidean Distance': [auc_euclidean, f1_euclidean, top_k_euclidean]
    })
    
    print(summary_table)
    
    return summary_table

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    from classification_task import classify_embeddings
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    knn_cosine_best, knn_euclidean_best = classify_embeddings(df)
    evaluate_metrics(df, knn_cosine_best, knn_euclidean_best)