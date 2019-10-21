'''SCRIPT TO CREATE TEST TRAIN SPLITS
name = "five_percent.csv"
id_path = "/media/olle/Seagate/kaggle/id_files/min_id_files_balanced"
save_path = "/media/olle/Seagate/kaggle/id_files/min_id_files_balanced/five_percent"
files_ = pd.read_csv(os.path.join(id_path,name))

files_shuffled = files_.sample(frac=1)[["image","level"]]
number_samples = files_shuffled.shape[0]
train = files_shuffled[0:int(number_samples*0.6)]
validation = files_shuffled[int(number_samples*0.6)+1:int(number_samples*0.8)]
test = files_shuffled[int(number_samples*0.8)+1:int(number_samples*1.0)]

train.to_csv(os.path.join(save_path,"train.csv"))
validation.to_csv(os.path.join(save_path,"validation.csv"))
test.to_csv(os.path.join(save_path,"test.csv"))
'''
