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
    def __init__(self, ft_paths, ae_paths, uae_paths):
        self.annotated_patients = [i.split("/")[-1].split("_")[0] for i in ae_paths]

        # set paths to instances
        self.feature_table_paths = ft_paths
        self.ae_paths = ae_paths
        self.uae_paths = uae_paths

        # load feature table
        self.features_pd = self.selection_table()

    def selection_table(self):
        """
        @return: joined feature table dataframes
        @rtype: DataFrame
        """
        feature_table = pd.DataFrame(columns = FEATURE_COLUMNS)
        for path in self.feature_table_paths:
            table = pd.read_csv(path, index_col = 0)
            feature_table = feature_table.append(table)
        return feature_table

    def filter_paths(self, paths, sampling_rate, filter_=True):
        filtered_pd = pd.DataFrame(columns = self.features_pd.columns.tolist() + ["embedding_path"])
        for p in tqdm(paths):
            # get record identifiers
            record = p.split("/")[-1]
            patient_id = record.split("_")[0]
            study_date = record.split("_")[1]
            laterality = record.split("_")[2]

            # extract boolean vector for record
            bool_ = ((self.features_pd.patient_id.astype(str) == patient_id) &
                     (self.features_pd.laterality == laterality) &
                     (self.features_pd.study_date.astype(str) == study_date))

            # filter for record
            record_statistic = self.features_pd.loc[bool_]

            # get statistics for embedded oct's
            idx = np.round(np.linspace(0, record_statistic.shape[0] - 1, sampling_rate)).astype(int)

            if record_statistic.shape[0] == 0:
                print("statistic table empty for record, moving to next sample")
                continue
            record_statistic = record_statistic.iloc[idx]

            # exclude any paths from already annotated patients
            if patient_id in self.annotated_patients:
                print(f"patient {patient_id} already annotated")

            if patient_id not in self.annotated_patients:
                if filter_:
                    srhm = record_statistic["4"]
                    drusen = record_statistic["8"]
                    fibPED = record_statistic["7"]
                    fibrosis = record_statistic["13"]

                    feature_bool = fibrosis > 50

                    # remove paths without feature of interest
                    filtered_record_statistic = record_statistic.loc[feature_bool]

                    # append embedding path
                    filtered_record_statistic["embedding_path"] = p
                    filtered_pd = filtered_pd.append(filtered_record_statistic.drop_duplicates())
                    continue
                else:
                    # append b scans with foi to data frame
                    record_statistic["embedding_path"] = p
                    filtered_pd = filtered_pd.append(record_statistic)
        return filtered_pd


'''
class Select(Filter):
    def __init__(self, budget, ft_path, ae_paths, uae_paths):
        self.budget = budget
        super().__init__(ft_path, ae_paths, uae_paths)

    def select_batch(self, embeddings, budget):
        kcenters = kCenterGreedy(embeddings)
        # select new indices
        [ind_to_label, min_dist] = kcenters.select_batch_(already_selected = kcenters.already_selected, N = budget)
        return [ind_to_label, min_dist]
'''

if __name__ == "__main__":
    file_manager = FileManager("annotated_files.csv")

    # get record paths
    unannotated_paths, annotated_paths = file_manager.unannotated_records(use_cache = False)
    unannotated_paths = random.sample(unannotated_paths, args.number_to_search)

    filter = Filter(file_manager.feature_table_paths, annotated_paths, unannotated_paths)

    features_table = filter.selection_table()
    features_filtered_pd = filter.filter_paths(filter.uae_paths, args.sampling_rate, filter_ = True)

    pprint(features_table.head(5))
    pprint(features_filtered_pd.head(5))

    print("number of unfiltered samples are:", features_table.shape)
    print("number of filtered samples are:", features_filtered_pd.shape)

    assert np.sum(features_filtered_pd.patient_id.isin(filter.annotated_patients)) == 0, \
        "allready selected patient choosen"
    assert features_table is not None, "returning None"
    assert features_filtered_pd is not None, "returning None"