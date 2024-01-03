import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from Model import Create_model
import os,sys
#from train import make_dataset
from rebuild_mols import convert_frags_to_smiles





format_dataset = lambda x,y,z: ({"encoder_inputs":x,"decoder_inputs":y},z)
def make_dataset(data_samples,
                 batch_size=64):
    source_seq_vec = data_samples
    past_target_seq_vec = []
    target_seq_vec = []
    for seq in source_seq_vec:
        past_target_seq_vec.append([1]+seq)
        target_seq_vec.append(seq+[2])
    source_padd_array = keras.preprocessing.sequence.pad_sequences(source_seq_vec,maxlen=max_seq_length,dtype='int64', padding='post',)
    past_target_padd_array = keras.preprocessing.sequence.pad_sequences(past_target_seq_vec,maxlen=max_seq_length,dtype='int64', padding='post',)
    target_padd_array = keras.preprocessing.sequence.pad_sequences(target_seq_vec,maxlen=max_seq_length,dtype='int64', padding='post',)
    dataset = tf.data.Dataset.from_tensor_slices((source_padd_array,past_target_padd_array,target_padd_array))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset

def enc_one_hot(frags):
    one_hot = [dict_vocab[i] for i in frags]
    return one_hot

def generater(predictions,temperature ):
    temperature = temperature
    predictions = np.asanyarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    prob_seq_save = []
    for batch_seq in exp_preds:
        prob_seq = []
        for length_seq in batch_seq:
            pre_token = length_seq / np.sum(length_seq)
            prob_token = np.random.multinomial(1,pre_token,1)
            one_hot_token = np.argmax(prob_token)
            if one_hot_token == 2:
                break
            else:
                if one_hot_token not in [0,1,2,3]:
                    prob_seq.append(one_hot_token)
        prob_seq_save.append(prob_seq)
    return prob_seq_save

def get_key(val):
    for key, value in dict_vocab.items():
        if val == value:
            return key
def one_hot_cover_to_text(prob_seq_save,dict_vocab):
    prob_text_seq_save = []
    for seq in prob_seq_save:
        text_seq = []
        for token in seq:
            text_seq.append(get_key(token))
        #print(text_seq)
        prob_text_seq_save.append(text_seq)
    return prob_text_seq_save



###
with open("./JSON/dict_vocab_01.json") as f:
    dict_vocab = json.load(f)

###
embed_size = 512
units = 512
vocab_size = len(dict_vocab)
model = Create_model(vocab_size=vocab_size,embed_size=embed_size,units=units)
model.load_weights("Checkpoints/init_train_01/epoch_0.h5")

###

df = pd.read_json("./data/test_data_01.json")

frags_list = list(df["frags"])
one_hot_list = [enc_one_hot(i) for i in frags_list]

max_seq_length=20
dataset = make_dataset(one_hot_list,batch_size=1)


#####
temperature_range = []
temperature = 1.0
gap = 0.2
_end = 2.5
while temperature < _end:
    temperature_range.append(np.float32(temperature))
    temperature += gap
gen_epoch = 100

###

for temperature in temperature_range:
    save_path = f"./Generated_smiles_001/Temprature_{'%.2f'%temperature}"
    try:
        os.makedirs(save_path)
        print("makedir")
    except:
        print("The dir exist!")
    total_smiles_list=[]
    for epoch in range(gen_epoch):
        num_mols = 0
        out_smiles_list = []
        ###
        for inputs_batch,targets_batch in tqdm(dataset):
            predictions =model(inputs_batch)
            prob_seq_save = generater(predictions,temperature = temperature)
            prob_text_seq_save = one_hot_cover_to_text(prob_seq_save,dict_vocab)
            ###
            for frags in prob_text_seq_save:
                try:
                    smiles = convert_frags_to_smiles(frags)
                    num_mols += 1
                except:
                    smiles = "combo_error!"
                out_smiles_list.append(smiles)
            ###
        total_smiles_list.append(out_smiles_list)
        print(f"The number of generated mols was {num_mols} \n EPOCH_{epoch} was finished! ")
        ###
    data = {}
    for i in range(len(total_smiles_list)):
        data[f"epoch_{i}"] = total_smiles_list[i]
    df_out = pd.DataFrame(data)
    
    df_out.to_csv(f"{save_path}/gen_smiles.csv")
    print(f"{save_path}/gen_smiles.csv was saved!")

