import pickle
import pandas as pd

def load_and_preprocess_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    flattened_data = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                flattened_data.append([syndrome_id, subject_id, image_id, embedding])
    
    df = pd.DataFrame(flattened_data, columns=['syndrome_id', 'subject_id', 'image_id', 'embedding'])
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data('mini_gm_public_v0.1.p')
    print(df.head())