from pprint import pprint

import numpy as np
import umap
from sklearn import decomposition
import pandas as pd
from tqdm import tqdm


def load_scan(path):
    return np.load(path).flatten()


def load_volume(path):
    embedding_vol = np.load(path)
    scans = embedding_vol.reshape(embedding_vol.shape[0], -1)
    return scans


def apply_umap(feature_vector):
    # first run faster PCA before final UMAP
    try:
        n_components = min(len(feature_vector), 30)
        pca = decomposition.PCA(n_components = n_components)
        pca.fit(np.array(feature_vector))
        X = pca.transform(np.array(feature_vector))
        res = umap.UMAP(metric = 'correlation').fit_transform(X)
    except:
        res = np.array([])
    return res


def reduce_dim_annotated(embedding_paths):
    embeddings = []
    for ep in embedding_paths:
        embedding = load_scan(ep)

        # add loaded embeddings
        embeddings.append(embedding)
    print("--- annotated images embedded ---")
    return apply_umap(embeddings)


def reduce_dim_unannotated(table, chunk):
    umap_embeddings = pd.DataFrame(columns = ["id", "embedding"])
    embedding_chunks = np.array_split(table.embedding_path.drop_duplicates().tolist(), chunk)
    for ec in tqdm(embedding_chunks):
        print("completed another embedding chunk")
        embeddings = [[], []]
        for ep in ec:
            try:
                features_ep = table[table.embedding_path == ep]
                embedding = load_volume(ep)
                embedding_frames = features_ep.frame.tolist()

                # add all embeddings in volume
                for frame in embedding_frames:
                    # print(frame, ep)
                    # extract id and embedding array
                    id_ = features_ep[features_ep.frame == frame].id.iloc[0]
                    embedding_array = embedding[frame, :]

                    assert isinstance(id_, str), "id value must be string"
                    assert type(embedding_array) is not np.array, "embedding vector must be numpy array"

                    if embedding_array.size == 0:
                        print("embedding array is empty, skip record")
                        continue
                    embeddings[0].append(id_)
                    embeddings[1].append(embedding_array)
            except:
                print("File failed to load: ", ep)
        print("finished loading")
        umap_ = apply_umap(embeddings[1].copy())

        # if umap is empty, then continue
        if umap_.size == 0:
            continue

        umap_pd = pd.DataFrame([embeddings[0].copy(), umap_]).T.rename(columns = {0: "id", 1: "embedding"})
        umap_embeddings = umap_embeddings.append(umap_pd)

    print("--- unannotated images embedded ---")
    return umap_embeddings


if __name__ == "__main__":
    import os
    import random
    from pathlib import Path
    import sys

    path = Path(os.getcwd())
    sys.path.append(str(path.parent))
    sys.path.append(str(path.parent.parent))

    from file_manager import FileManager
    from filter import Filter
    from utils import args

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
    assert (features_ffiltered_pd.embedding_path.drop_duplicates().shape[0] // args.chunk_size) > 5 and not (
                args.chunk_size > 1), "chunk size to large"

    embedding = OCTEmbeddings()

    # embedding
    ua_embeddings = embedding.reduce_dim_unannotated(features_ffiltered_pd, chunk = args.chunk_size)

    assert embedding.reduce_dim_unannotated(pd.DataFrame(columns = features_ffiltered_pd.columns.values.tolist()),
                                            chunk = args.chunk_size).size == 0, "function does not handle empty DF"

    assert ua_embeddings.shape[0] == features_ffiltered_pd.shape[0], "not all filtered oct were embedded"
