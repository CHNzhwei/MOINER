import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class calculate_feature_similarity():

    def calculate_cosine_similarity(data_df):
        similarity = cosine_similarity(data_df.values.T)
        similarity = np.nan_to_num(similarity,copy=False)
        similarity_df = pd.DataFrame(similarity,index=data_df.columns.values,columns=data_df.columns.values)
        with h5py.File('./results_preprocessing/5.feature_similarity.h5','w') as f:
            f.create_dataset("feature_similarity",data=similarity_df)
        return similarity_df

    def calculate_euclidean_distances(data_df):
        similarity = euclidean_distances(data_df.values.T)
        similarity = np.nan_to_num(similarity,copy=False)
        similarity_df = pd.DataFrame(similarity,index=data_df.columns.values,columns=data_df.columns.values)
        with h5py.File('./results_preprocessing/5.feature_similarity.h5','w') as f:
            f.create_dataset("feature_similarity",data=similarity_df)
        return similarity_df