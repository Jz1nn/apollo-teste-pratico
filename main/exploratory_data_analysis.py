import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data(df):
    num_syndromes = df['syndrome_id'].nunique()
    print(f'Number of syndromes: {num_syndromes}')
    
    images_per_syndrome = df.groupby('syndrome_id').size()
    print(images_per_syndrome)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='syndrome_id', data=df, order=df['syndrome_id'].value_counts().index)
    plt.title('Distribution of number of images by syndrome')
    plt.xlabel('Syndrome')
    plt.ylabel('Number of images')
    plt.xticks(rotation=90)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Salvar a imagem na pasta 'results'
    plt.savefig(os.path.join(results_dir, 'images_by_syndrome.png'))

    plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    analyze_data(df)