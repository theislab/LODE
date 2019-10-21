import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
save_dir = "./resultplots"
image_net_file = "./output_20_imagenet_.csv"
random_net_file = "./output_20_random_.csv"
thickness_net_file = "./output_20_thickness_map_.csv"

plt.figure(figsize=(20,10))

imagenet_pd = pd.read_csv(image_net_file)
randomnet_pd = pd.read_csv(random_net_file)
thickness_pd = pd.read_csv(thickness_net_file)


plt.plot(randomnet_pd.loss,label="train random")
plt.plot(imagenet_pd.loss,label="train imagenet")
plt.plot(thickness_pd.loss,label="train thickness")

plt.plot(randomnet_pd.val_loss,label="val random",linestyle="--")
plt.plot(imagenet_pd.val_loss,label="val imagenet",linestyle="--")
plt.plot(thickness_pd.val_loss,label="val thickness",linestyle="--")


plt.title("loss")
plt.legend()
plt.savefig(os.path.join(save_dir,"loss_20.png"))
plt.close()

plt.figure(figsize=(20,10))
plt.plot(randomnet_pd.acc,label="train random")
plt.plot(imagenet_pd.acc,label="train imagenet")
plt.plot(thickness_pd.acc,label="train thickness")

plt.plot(randomnet_pd.val_acc,label="val random",linestyle="--")
plt.plot(imagenet_pd.val_acc,label="val imagenet",linestyle="--")
plt.plot(thickness_pd.val_acc,label="val thickness",linestyle="--")


plt.title("accuracy")
plt.legend()
plt.savefig(os.path.join(save_dir,"accuracy_20.png"))

plt.close()

plt.figure(figsize=(20,10))
plt.plot(randomnet_pd.lr,label="random")
plt.plot(imagenet_pd.lr,label="imagenet")
plt.plot(thickness_pd.lr,label="thickness")


plt.title("learning rate")
plt.legend()
plt.savefig(os.path.join(save_dir,"learning_rate_20.png"))