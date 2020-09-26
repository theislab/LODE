from pprint import pprint
import pandas as pd
import os
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from kcenter_greedy_nalu import kCenterGreedy

class Select():
    def __init__(self, budget):
        self.budget = budget

    def select_batch(self, embeddings):
        kcenters = kCenterGreedy(embeddings)
        # select new indices
        [ind_to_label, min_dist] = kcenters.select_batch_(already_selected = kcenters.already_selected, N = self.budget)
        return [ind_to_label, min_dist]


if __name__ == "__main__":
    import os
    from pathlib import Path
    import sys

    path = Path(os.getcwd())
    sys.path.append(str(path.parent))
    sys.path.append(str(path.parent.parent))

    # add children paths
    for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
        sys.path.append(str(child_dir))

    from file_manager import FileManager
    from filter import Filter
    from embeddings import OCTEmbeddings
    from utils import args, move_selected_octs
    from kcenter_greedy_nalu import kCenterGreedy
    from config import WORK_SPACE, EMBEDD_DIR

    file_manager = FileManager("annotated_files.csv")

    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)

    assert args.number_to_search <= unannotated_pd.shape[0], "searching more images than exist filtered"
    unannotated_pd = unannotated_pd.sample(args.number_to_search)

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

    selection = Select(args.budget)
    [ind_to_label, min_dist] = selection.select_batch(ua_embeddings)

    selected_scans = ua_embeddings.iloc[ind_to_label]

    print("format csv")
    selected_scans_pd = selected_scans.id.str.split("_", expand = True).rename(
        columns = {0: "patient_id", 1: "study_date", 2: "laterality", 3: "frame"})

    # assign id
    selected_scans_pd["id"] = selected_scans["id"]

    # add dicom name
    selected_scans_pd = pd.merge(selected_scans_pd, features_ffiltered_pd[["dicom", "id"]],
                                 how = "left", left_on = "id", right_on = "id")

    print("records to select for annotations are: ", selected_scans)
    DST_DIR = os.path.join(WORK_SPACE, "active_learning", f"selected_{args.name}")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    assert selected_scans_pd.patient_id.drop_duplicates().shape[0] == selected_scans_pd.shape[0], \
        "patients selected are not unique"

    selected_path = os.path.join(DST_DIR, f"records_selected_{args.name}.csv")
    selected_scans_pd.to_csv(selected_path)
