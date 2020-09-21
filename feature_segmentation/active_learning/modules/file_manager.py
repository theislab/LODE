import os
import glob
import pandas as pd
import random
from feature_segmentation.config import WORK_SPACE, EMBEDD_DIR


class FileManager:
    def __init__(self, annotated_file):
        self.annotated_file = annotated_file
        self.cache_dir = os.path.join(WORK_SPACE, "feature_segmentation/active_learning/cache")

    @property
    def feature_table_paths(self):
        return glob.glob(os.path.join(WORK_SPACE, "segmentation/feature_tables/*"))

    @property
    def annotated_patients(self):
        embeddings = pd.read_csv(os.path.join(WORK_SPACE, f"feature_segmentation/active_learning/{self.annotated_file}"),
                                    ).dropna()["0"].tolist()

        # extract patient ids from first column
        ids = list(map(lambda x: str(x).split("_")[0], embeddings))

        # return only valid numerical patient ids
        return list(filter(lambda x: x.isdigit(), ids))

    def unannotated_records(self, use_cache=False):
        """
        @param use_cache: boolean to indicate whether to use chached files or not
        @type use_cache: bool
        @return: file paths of unannotated files
        @rtype: list
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if use_cache:
            unannotated_paths = pd.read_csv(os.path.join(self.cache_dir, "unannotated_paths.csv"))["0"].tolist()
        else:
            unannotated_paths = glob.glob(EMBEDD_DIR + "/*")
            pd.DataFrame(unannotated_paths).to_csv(os.path.join(self.cache_dir, "unannotated_paths.csv"), index = 0)
        return unannotated_paths

    def get_annotated_embedding_paths(self, unannotated_paths):
        annotated_paths = list(filter(lambda x: x.split("/")[-1].split("_")[0] in self.annotated_patients,
                                      unannotated_paths))
        return annotated_paths


if __name__ == "__main__":
    file_manager = FileManager("annotated_files.csv")

    class Args():
        def __init__(self, number_to_search):
            self.number_to_search = number_to_search

    args = Args(number_to_search = 10)

    # get record paths
    unannotated_embeddings_paths = file_manager.unannotated_records(use_cache = True)
    annotated_embeddings_paths = file_manager.get_annotated_embedding_paths(unannotated_embeddings_paths)
    unannotated_embeddings_paths = random.sample(unannotated_embeddings_paths, args.number_to_search)

    print("number of embedded volumes", len(unannotated_embeddings_paths))
    print("number of annotated embedded volumes", len(annotated_embeddings_paths))
