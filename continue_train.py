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
                 max_seq_length = None,
                 batch_size=None):
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


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()
def train_step(inputs,targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs,training = True)
        #pre = tf.argmax(predictions,axis=-1)
        #print(pre,pre.shape)
        loss = loss_fn(targets,predictions)
        #print(targets)
        #print(predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()
        #print(logs)

    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()
    return logs

def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()




with open("./JSON/dict_vocab.json") as f:
    dict_vocab = json.load(f)
embed_size = 512
units = 512
vocab_size = len(dict_vocab)
model = Create_model(vocab_size=vocab_size,embed_size=embed_size,units=units)
model.load_weights(f"Checkpoints/init_train/epoch_4.h5")


epochs = 5
df = pd.read_json("./data/test_data.json")


for epoch in  range(epochs):
    reset_metrics()


    df = df.sample(frac=1)
    df.to_json(f"./data/test_data_{epoch}.json")
    frags_list = list(df["frags"])
    one_hot_list = [enc_one_hot(i) for i in frags_list]
    max_seq_length = 20
    batch_size = 5
    train_dataset = make_dataset(one_hot_list,batch_size=batch_size,max_seq_length =max_seq_length)


    train_metric = []
    train_loss = []
    for inputs_batch,targets_batch in tqdm(train_dataset):
        logs = train_step(inputs_batch,targets_batch)
        #logs['sparse_categorical_accuracy']
        train_metric.append(logs['sparse_categorical_accuracy'].numpy())
        train_loss.append(logs['loss'].numpy())
        #val_acc = []
        #mean_val_acc = #val_acc_total.append(mean_val_acc)
    print(f"Results at the end of epoch {epoch}")
    for key, value in logs.items() :
        print(f"...{key}: {value:.4f}")
    df_met = pd.DataFrame({"acc":train_metric,"loss":train_loss})
    df_met.to_csv(f"./Checkpoints/continue_train/contitu_train_metric_{epoch}.csv")
    model.save_weights(f"./Checkpoints/continue_train/epoch_{epoch}.h5")
