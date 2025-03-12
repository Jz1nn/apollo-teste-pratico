# Apollo Project Analysis

## Summary
1. Context
2. Challenge
3. Solution Development
4. Conclusion and Demonstration
5. Next Steps

# **Context**
- The Apollo project involves the analysis of a dataset containing image embeddings associated with different syndromes. The goal is to explore and classify these images using machine learning techniques.

# **Challenge**
## Problem
- Apply data analysis and machine learning techniques to classify images of different syndromes based on their embeddings.

## Causes
- Need to better understand image data and identify patterns and trends.
- Need to develop predictive models to improve syndrome classification.

## Solution
- Use exploratory data analysis (EDA) techniques to gain insights.
- Develop machine learning models, such as K-Nearest Neighbors (KNN), for syndrome classification.

# **Solution Development and Business Insights**
1. Data Description
- The dataset contains image embeddings associated with different syndromes.
2. Descriptive Statistics
- Analysis of the distribution of images by syndrome.
3. Exploratory Data Analysis (EDA)
- Verification of data integrity and treatment of missing values.
- Analysis of imbalance in the number of images by syndrome.
- Visualization of embeddings using t-SNE for dimensionality reduction.
4. Machine Learning Modeling
- Implementation of K-Nearest Neighbors (KNN) with different distance metrics (cosine and euclidean).
- Evaluation of models using cross-validation and metrics such as AUC, F1-score and Top-k accuracy.

# **Conclusion and Demonstration**
### Model Results
- The cosine distance metric performed better compared to the euclidean metric.
- Cluster analysis using t-SNE showed a clear separation between some syndromes, indicating that embeddings are effective for classification.
![embeddings](/results/visualization_of_embeddings.png)

### Visualization and Analysis
- Image distribution plots by syndrome.
- t-SNE visualization of embeddings.
- ROC AUC curves for cosine and euclidean distance metrics.

## Model Performance Comparison
- The cosine distance metric showed higher accuracy and AUC compared to the euclidean metric.
![roc_curve](/results/roc_curve_comparison.png)

# **Next Steps**
1. Refine machine learning models to improve accuracy.
2. Explore other dimensionality reduction techniques and classification algorithms. 3. Integrate the results with a generative AI system to generate insights in natural language.

## Final Considerations
This project allowed us to explore advanced data analysis and machine learning techniques, providing valuable insights into the classification of syndrome images. The integration of visualization and model evaluation techniques was essential to understand the performance of the algorithms and identify areas for improvement.

### Author: John Willian de Jesus Soares
### Date: 02/03/2025
### Contact: jz1nnwln@gmail.com
### Linkedin: www.linkedin.com/in/jz1nnwln/