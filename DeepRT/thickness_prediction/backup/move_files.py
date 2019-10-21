import shutil
import os

src_f = "/media/olle/Seagate/thickness_map_prediction/fundus"
dest_f = "/media/olle/Seagate/thickness_map_prediction/thickness_maps"

src_files = os.listdir(src_f)

for sf in src_files:
    if ".npy" in sf:
        shutil.mv(os.path.join(src_f,sf),os.path.join(dest_f,src_f))