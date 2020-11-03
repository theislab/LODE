import os
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from kcenter_greedy_nalu import kCenterGreedy


def select_batch(embeddings, budget):
    kcenters = kCenterGreedy(embeddings)
    # select new indices
    [ind_to_label, min_dist] = kcenters.select_batch_(already_selected = kcenters.already_selected, N = budget)
    return [ind_to_label, min_dist]
