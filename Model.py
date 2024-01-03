import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Encoder(keras.models.Model):
    def __init__(self,vocab_size,embed_size,units,**kwargs):
        super().__init__(**kwargs)
        self.embed_size=embed_size
        self.units=units
        self.embedding_layer = layers.Embedding(input_dim=vocab_size,output_dim=embed_size,mask_zero=True)
        self.lstm_layer = layers.LSTM(units,return_sequences=True,return_state=True)
        #self.brnn_lstm = layers.Bidirectional(self.lstm_layer,merge_mode="concat")

    def call(self,inputs):
        embed = self.embedding_layer(inputs)
        #brnn_output = self.brnn_lstm(embed)
        #state_h = brnn_output[1]
        #state_c = brnn_output[2]
        encoder_output,state_h,state_c = self.lstm_layer(embed)
        encoder_state = [state_h,state_c]
        return encoder_state
class Decoder(keras.models.Model):
    def __init__(self,vocab_size,embed_size,units,**kwargs):
        super().__init__(**kwargs)
        self.embed_size=embed_size
        self.units=units
        self.embedding_layer = layers.Embedding(input_dim=vocab_size,output_dim=embed_size,mask_zero=True)
        self.lstm_layer = layers.LSTM(units,return_sequences=True,return_state=True)

    def call(self,inputs,state):
        embed = self.embedding_layer(inputs)
        decoder_output,final_memory_state, final_carry_state = self.lstm_layer(embed,initial_state=state)
        return decoder_output
def Create_model(vocab_size,embed_size,units):
    encoder_inputs = keras.layers.Input(shape=[None],name='encoder_inputs')
    decoder_inputs = keras.layers.Input(shape=[None],name='decoder_inputs')
    encoder_state = Encoder(vocab_size = vocab_size,embed_size=embed_size,units=units, name = "encoder")(encoder_inputs)
    decoder_output = Decoder(vocab_size = vocab_size,embed_size=embed_size,units=units,name = "decoder")(decoder_inputs,encoder_state)
    output = keras.layers.Dense(vocab_size,activation='softmax',name='dense')(decoder_output)
    model = keras.Model([encoder_inputs, decoder_inputs], output)
    return model
