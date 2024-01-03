import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from Model import Create_model


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

with open("./JSON/dict_vocab.json") as f:
    dict_vocab = json.load(f)
df = pd.read_json("./data/val_data.json")
frags_list = list(df["frags"])
one_hot_list = [enc_one_hot(i) for i in frags_list]

max_seq_length = 20
batch_size = 128
val_dataset = make_dataset(one_hot_list,batch_size=batch_size)

embed_size = 512
units = 512
vocab_size = len(dict_vocab)
model = Create_model(vocab_size=vocab_size,embed_size=embed_size,units=units)


val_metric_total={}
acc_metric = keras.metrics.SparseCategoricalAccuracy()
m = tf.keras.metrics.Mean()
epochs = 5
for epoch in range(epochs):
    model.load_weights(f"Checkpoints/init_train/epoch_{epoch}.h5")
    val_metric = []
    for inputs_batch,targets_batch in tqdm(val_dataset):
        predictions = model(inputs_batch,training = False)
        acc = acc_metric(targets_batch,predictions)

        mean_acc = m(acc)
        mean_acc = mean_acc.numpy()
        val_metric.append(mean_acc)
    val_metric_total.update({f"epoch_{epoch}":val_metric})
df_val_acc = pd.DataFrame(val_metric_total)

df_val_acc.to_csv("./Checkpoints/init_train/val.csv")


