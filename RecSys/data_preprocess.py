from collections import Counter
import numpy as np
import random as rd
import pandas as pd

def filter_data(data, min_inter):
    count_i = Counter(list(data["movie_id"]))
    items = list(set([k for k, v in count_i.items() if v >= min_inter]))
    data = data.loc[(data["movie_id"].isin(items))]

    count_u = Counter(list(data["user_id"]))
    users = list(set([k for k, v in count_u.items() if v >= min_inter]))

    filtered_data = data.loc[(data["user_id"].isin(users))]
    return filtered_data

def remap_id(list_old_id):
    new_dict = {}
    for i in range (len(list_old_id)):
        new_dict[list_old_id[i]] = i
    return new_dict
    
def remap_data(data):
    users = data["user_id"].unique().tolist()
    items = data["movie_id"].unique().tolist()
    dict_new_uid = remap_id(users)
    dict_new_iid = remap_id(items)
    
    new_uid_col = []
    new_iid_col = []
    for i  in range (data.shape[0]):
        row = data.iloc[i]
        new_uid_col.append(dict_new_uid[row["user_id"]])
        new_iid_col.append(dict_new_iid[row["movie_id"]])
    data["uid"] = new_uid_col
    data["iid"] = new_iid_col
    return data
        
def split_train_test(data):
    data = data.sample(frac=1)
    data_size = data.shape[0]

    train = data[0: int(0.8*data_size)]
    valid = data[int(0.8*data_size) : int(0.9*data_size)]
    test = data[int(0.9*data_size) : data_size]  
    return train, valid, test  


if __name__ == '__main__':

    data = pd.read_csv("../Data_Preprocessing/Preprocessed_Data/ratings.csv")
    filtered_data = filter_data(data, 20)
    remaped_data = remap_data(filtered_data)
    train, valid, test = split_train_test(remaped_data)
    train.to_csv("data/train.csv", index=False)
    valid.to_csv("data/valid.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    
