from pprint import pprint

import numpy as np
import umap
from sklearn import decomposition
import pandas as pd
from tqdm import tqdm
import os
import random
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from file_manager import FileManager
from utils import args

FEATURE_COLUMNS = ['patient_id', 'study_date', 'laterality', '0', '1', '10', '11', '12',
                   '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']


class OCTEmbeddings:
    def __init__(self):
        super()

    @staticmethod
    def load_scan(path):
        return np.load(path).flatten()

    @staticmethod
    def load_volume(path):
        embedding_vol = np.load(path)
        scans = embedding_vol.reshape(embedding_vol.shape[0], -1)
        return scans

    @staticmethod
    def apply_umap(feature_vector):
        # first run faster PCA before final UMAP
        try:
            n_components = min(len(feature_vector), 30)
            pca = decomposition.PCA(n_components = n_components)
            pca.fit(np.array(feature_vector))
            X = pca.transform(np.array(feature_vector))
            res = umap.UMAP(metric = 'correlation').fit_transform(X)
        except:
            res = None
        return res

    def reduce_dim_annotated(self, embedding_paths):
        embeddings = []
        for ep in embedding_paths:
            embedding = self.load_scan(ep)

            # add loaded embeddings
            embeddings.append(embedding)
        print("--- annotated images embedded ---")
        return self.apply_umap(embeddings)

    def reduce_dim_unannotated(self, table, chunk):
        umap_embeddings = pd.DataFrame(columns = ["id", "embedding"])
        embedding_chunks = np.array_split(table.embedding_path.drop_duplicates().tolist(), chunk)
        for ec in tqdm(embedding_chunks):
            print("completed another embedding chunk")
            embeddings = [[], []]
            for ep in ec:
                features_ep = table[table.embedding_path == ep]
                embedding = self.load_volume(ep)
                embedding_frames = np.round(np.linspace(0, 48, embedding.shape[0]))

                # add all embeddings in volume
                for iter_ in features_ep.frame.tolist():
                    embed_loc = np.where(embedding_frames == iter_)[0][0]

                    # extract id and embedding array
                    id_ = features_ep[features_ep.frame == iter_].id.iloc[0]
                    embedding_array = embedding[embed_loc, :]

                    assert isinstance(id_, str), "id value must be string"
                    assert type(embedding_array) is not np.array, "embedding vector must be numpy array"

                    if embedding_array.size == 0:
                        print("embedding array is empty, skip record")
                        continue

                    embeddings[0].append(id_)
                    embeddings[1].append(embedding_array)

            umap_ = self.apply_umap(embeddings[1].copy())
            if not umap_:
                continue

            umap_embeddings = umap_embeddings.append(
                pd.DataFrame([embeddings[0].copy(), umap_]).T.rename(columns = {0: "id", 1: "embedding"}))
        print("--- unannotated images embedded ---")
        return umap_embeddings


class Filter():
    def __init__(self, ft_paths, uae_paths):
        # set paths to instances
        self.feature_table_paths = ft_paths
        self.uae_paths = uae_paths

    def selection_table(self):
        """
        @return: joined feature table dataframes
        @rtype: DataFrame
        """
        feature_table = pd.DataFrame(columns = FEATURE_COLUMNS)
        for path in self.feature_table_paths:
            table = pd.read_csv(path, index_col = 0)
            feature_table = feature_table.append(table)

        feature_table.study_date = feature_table.study_date.astype(str)
        feature_table.patient_id = feature_table.patient_id.astype(str)
        return feature_table

    def filter_paths(self, features_table):
        """
        @param features_table:
        @type features_table:
        @return:
        @rtype:
        """
        fibrosis_bool = features_table["13"] > 50
        features_table_filtered = features_table[fibrosis_bool]
        return features_table_filtered


if __name__ == "__main__":
    file_manager = FileManager("annotated_files.csv")

    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)
    # unannotated_pd = unannotated_pd.sample(args.number_to_search)

    filter = Filter(file_manager.feature_table_paths, unannotated_pd)

    features_table = filter.selection_table()

    keys = ["patient_id", "laterality", "study_date"]
    features_table_pd = pd.merge(unannotated_pd, features_table, left_on = keys, right_on = keys, how = "left")
    features_ffiltered_pd = filter.filter_paths(features_table_pd)

    pprint(features_table.head(5))
    pprint(features_ffiltered_pd.head(5))

    print("number of unfiltered samples are:", features_table.shape)
    print("number of filtered samples are:", features_ffiltered_pd.shape)

    assert sum(unannotated_pd.patient_id.isin(annotated_pd.patient_id.values)) == 0, "patient overlap"
    assert sum(features_ffiltered_pd["13"] < 50) == 0, "all record contains feature oi"
    assert features_table is not None, "returning None"
    assert features_ffiltered_pd is not None, "returning None"