import glob

import numpy as np
import umap
from sklearn import decomposition
import pandas as pd
import shutil
from pydicom import read_file
import cv2
from tqdm import tqdm
from active_learning.kcenter_greedy_nalu import kCenterGreedy
from active_learning.config import OCT_DIR, WORK_SPACE, EMBEDD_DIR
import os

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
        pca = decomposition.PCA(n_components = 30)
        pca.fit(np.array(feature_vector))
        X = pca.transform(np.array(feature_vector))
        return umap.UMAP(metric = 'correlation').fit_transform(X)

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
                    id = features_ep[features_ep.frame == iter_].id.iloc[0]
                    embedding_array = embedding[embed_loc, :]

                    assert isinstance(id, str), "id value must be string"
                    assert type(embedding_array) is not np.array, "embedding vector must be numpy array"

                    embeddings[0].append(id)
                    embeddings[1].append(embedding_array)

            umap_ = self.apply_umap(embeddings[1].copy())
            umap_embeddings = umap_embeddings.append(
                pd.DataFrame([embeddings[0].copy(), umap_]).T.rename(columns = {0: "id", 1: "embedding"}))
        print("--- unannotated images embedded ---")
        return umap_embeddings


class OctOI(OCTEmbeddings):
    def __init__(self, ft_paths, ae_paths, uae_paths):
        self.annotated_patients = [i.split("/")[-1].split("_")[0] for i in ae_paths]

        # load features table
        self.feature_table_paths = ft_paths
        self.features_pd = self.selection_table()

        self.ae_paths = ae_paths
        self.uae_paths = uae_paths

    def selection_table(self):
        """
        :return:
        :rtype:
        """
        feature_table = pd.DataFrame(columns = FEATURE_COLUMNS)
        for path in self.feature_table_paths:
            table = pd.read_csv(path, index_col = 0)
            # table = table.groupby(["patient_id", "study_date", "laterality"]).sum()
            # table.reset_index(inplace = True)
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

                    feature_bool = ((srhm > 50) | (drusen > 50) | (fibPED > 50) | (fibrosis > 50))

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


class Select(OctOI):
    def __init__(self, budget, ft_path, ae_paths, uae_paths):
        self.budget = budget
        super().__init__(ft_path, ae_paths, uae_paths)

    def select_batch(self, embeddings, budget):
        kcenters = kCenterGreedy(embeddings)
        # select new indices
        [ind_to_label, min_dist] = kcenters.select_batch_(already_selected = kcenters.already_selected, N = budget)
        return [ind_to_label, min_dist]


class FileManager:
    def __init__(self, annotated_file):
        self.annotated_file = annotated_file

    @property
    def feature_table_paths(self):
        return glob.glob(os.path.join(WORK_SPACE, "segmentation/feature_tables/*"))

    @property
    def annotated_patients(self):
        embeddings = pd.read_csv(os.path.join(WORK_SPACE, f"active_learning/{self.annotated_file}"),
                                 header = None).dropna()[0].tolist()

        # extract patient ids from first column
        ids = list(map(lambda x: str(x).split("_")[0], embeddings))

        # return only valid numerical patient ids
        return list(filter(lambda x: x.isdigit(), ids))

    @property
    def unannotated_records(self):
        return glob.glob(EMBEDD_DIR + "/*")

    def get_annotated_embedding_paths(self):
        annotated_paths = list(filter(lambda x: x.split("/")[-1].split("_")[0] in self.annotated_patients,
                                      self.unannotated_records))
        return annotated_paths


def to_three_channel(img):
    return np.stack((img,) * 3, axis = -1)


def move_selected_octs(selected_pd, dst_dir):
    dicom_paths = []
    for row in selected_pd.itertuples():
        dicom_file_path = os.path.join(OCT_DIR, str(row.patient_id), row.laterality, str(row.study_date), row.dicom)
        # dicom_file_path = os.path.join(OCT_DIR, row.dicom)
        # load dicom file if not empty
        dc = read_file(dicom_file_path)
        vol = dc.pixel_array
        oct_ = vol[int(row.frame), :, :]

        if len(oct_.shape) == 2:
            oct_ = to_three_channel(oct_)

        record_name = f"{row.patient_id}_{row.laterality}_{row.study_date}"
        oct_name = f"{row.patient_id}_{row.laterality}_{row.study_date}_{row.frame}"
        oct_dst_dir = os.path.join(dst_dir, record_name)
        dicom_paths.append(dicom_file_path)

        # create dir for selected record
        os.makedirs(oct_dst_dir, exist_ok = True)

        # copy selected oct
        cv2.imwrite(os.path.join(oct_dst_dir, oct_name + ".png"), oct_)

        # copy oct volume
        os.makedirs(os.path.join(oct_dst_dir, "vol"), exist_ok = True)
        for j in range(vol.shape[0]):
            oct_slice = vol[j, :, :]

            # convert to 3 channel if necessary
            if len(oct_slice.shape) == 2:
                oct_slice = to_three_channel(oct_slice)
            cv2.imwrite(os.path.join(os.path.join(oct_dst_dir, "vol"), f"{record_name}_{j}.png"), oct_slice)

    # save list to data frame
    pd.DataFrame(dicom_paths).to_csv(os.path.join(dst_dir, "dicom_paths.csv"))
