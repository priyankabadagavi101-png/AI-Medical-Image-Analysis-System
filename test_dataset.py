import os

train_path = "data/chest_xray/train"

for category in os.listdir(train_path):
    folder = os.path.join(train_path, category)
    print(category, ":", len(os.listdir(folder)), "images")