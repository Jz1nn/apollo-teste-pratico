import warnings
from sklearn.exceptions import UndefinedMetricWarning
from data_preprocessing import load_and_preprocess_data
from exploratory_data_analysis import analyze_data
from data_visualization import visualize_embeddings
from classification_task import classify_embeddings
from metrics_evaluation import evaluate_metrics
from report import generate_report

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    print("\nLoading and preprocessing data...\n")
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    
    print("\nAnalyzing data...\n")
    analyze_data(df)
    
    print("\nVisualizing embeddings...\n")
    visualize_embeddings(df)
    
    print("\nClassifying embeddings...\n")
    knn_cosine_best, knn_euclidean_best = classify_embeddings(df)
    
    print("\nEvaluating metrics...\n")
    summary_table = evaluate_metrics(df, knn_cosine_best, knn_euclidean_best)
    
    print("\nGenerating report...\n")
    generate_report(summary_table)