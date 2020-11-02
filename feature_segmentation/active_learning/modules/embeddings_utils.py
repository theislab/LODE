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

        print("finished loading")
        umap_ = apply_umap(embeddings[1].copy())

        # if umap is empty, then continue
        if umap_.size == 0:
            continue

        umap_pd = pd.DataFrame([embeddings[0].copy(), umap_]).T.rename(columns = {0: "id", 1: "embedding"})
        umap_embeddings = umap_embeddings.append(umap_pd)

    print("--- unannotated images embedded ---")
    return umap_embeddings