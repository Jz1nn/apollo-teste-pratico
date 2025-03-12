def generate_report(summary_table):
    auc_cosine = summary_table.loc[summary_table['Metric'] == 'AUC', 'Cosine Distance'].values[0]
    f1_cosine = summary_table.loc[summary_table['Metric'] == 'F1-score', 'Cosine Distance'].values[0]
    top_k_cosine = summary_table.loc[summary_table['Metric'] == 'Top-k Accuracy', 'Cosine Distance'].values[0]
    auc_euclidean = summary_table.loc[summary_table['Metric'] == 'AUC', 'Euclidean Distance'].values[0]
    f1_euclidean = summary_table.loc[summary_table['Metric'] == 'F1-score', 'Euclidean Distance'].values[0]
    top_k_euclidean = summary_table.loc[summary_table['Metric'] == 'Top-k Accuracy', 'Euclidean Distance'].values[0]

    report = f"""
    # Genetic Syndrome Classification Analysis Report

    ## Methodology

    ### Data Preprocessing
    1. Data Loading: The data was loaded from a pickle file.
    2. Data Structure Flattening: The hierarchical structure of the data was flattened to a format 
    suitable for analysis.
    3. Integrity Check: We checked the integrity of the data and treated missing or inconsistent data.

    ### Algorithm Selection and Parameter Selection
    1. K-Nearest Neighbors (KNN) Algorithm: We used the KNN algorithm to classify the embeddings.
    2. Distance Metrics: We evaluated two distance metrics: cosine and Euclidean.
    3. Cross Validation: We performed 10-fold cross validation to determine the optimal value of k 
    (from 1 to 15).

    ## Results

    ### Model Performance
    We present the findings with supporting graphs and tables.

    #### ROC AUC Curves
    ![ROC Curve Comparison](roc_curve_comparison.png)

    #### Performance Metrics Table
    | Metric          | Cosine Distance | Euclidean Distance |
    |-----------------|-----------------|--------------------|
    | AUC             | {auc_cosine:.3f}           | {auc_euclidean:.3f}              |
    | F1-Score        | {f1_cosine:.3f}           | {f1_euclidean:.3f}              |
    | Top-k Accuracy  | {top_k_cosine:.3f}           | {top_k_euclidean:.3f}              |

    ## Analysis

    ### Data Processing
    - Observed that some syndromes have significantly more images than others.
    - This may indicate an imbalance in the dataset, which may affect the performance of classification models.
    - Syndromes with fewer images may be more difficult to classify correctly.

    ### Data Visualization
    - In the t-SNE visualization, we can observe the formation of distinct clusters for different syndromes.
    - These clusters indicate that the embeddings of images of similar syndromes are close together in the embedding space.
    - This suggests that the classification model may be able to distinguish between different syndromes based on the embeddings.

    - How the patterns relate to the classification task:
    - The clusters observed are important for the classification task because they indicate that there is a separation between some syndromes.
    - If the clusters are well defined, the classification model KNN, can perform well in classifying the syndromes.
    - However, if there is overlap between the clusters, this may indicate that some syndromes are more difficult to distinguish and may require more advanced classification techniques.
    
    ### Performance Comparison
    - The cosine distance metric showed higher accuracy and AUC compared to the Euclidean distance metric.
    - This suggests that the cosine distance is better suited for this classification task.
    - The F1 score and Top-k accuracy also indicate that the cosine distance metric performs better overall.

    ### Challenges and Solutions
    - I had difficulty handling the 4th task because the "roc_curve" function expected 
    "y_true" to contain binary values (0 or 1) for each class, but "y_true" contained 
    multiple classes.
    - The solution was to binarize "y_true" for each class separately and calculate the 
    ROC curve for each of them.

    ## Recommendations

    ### Potential Improvements
    - Explore other distance metrics or more advanced classification algorithms.
    - Increase the dataset size to improve the robustness of the model.

    ### Next Steps
    - Implement data augmentation techniques to address the imbalance in the dataset.
    - Perform further analysis of classification errors to identify potential improvements
    to the model.
    """
    print(report)

if __name__ == "__main__":
    from metrics_evaluation import evaluate_metrics
    from data_preprocessing import load_and_preprocess_data
    from classification_task import classify_embeddings

    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    knn_cosine_best, knn_euclidean_best = classify_embeddings(df)
    summary_table = evaluate_metrics(df, knn_cosine_best, knn_euclidean_best)
    
    generate_report(summary_table)