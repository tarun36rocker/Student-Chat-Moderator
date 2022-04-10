from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('accuracy')>0.85):
        print("\n 85% acc reached")
        self.model.stop_training = True

''' #using wget to download json file
import wget
#downloading zip folder using wget
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
filename = wget.download(url)
print(filename)'''

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("C:/Users/Tarun/Desktop/comp/pycharm/NLP/chatbot.json", 'r') as f:
    datastore = json.load(f)
#print(datastore)
sentences= []
labels = []


for item in datastore:
    sentences.append(item['chat'])
    labels.append(item['is_irrelevent'])

#spliting training and testing set
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#Tokenization is a method to segregate a particular text into small chunks or tokens. Here the tokens or chunks can be anything from words to characters, even subwords.
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) #num_words -> num of tokens , Out Of Vocab token - this will replace any unknown words with a token of our choosing
tokenizer.fit_on_texts(training_sentences) #fits tokenizer on the sentences

word_index = tokenizer.word_index # assigns each unique word with a number
#print(word_index)  #'bb': 22255, "italy's": 22256, 'etna': 22257

training_sequences = tokenizer.texts_to_sequences(training_sentences) #converts each sentence or list of words into their numbers from word_index
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type) #padding or adding post padding of 0's to the end to mantain uniformity between each sentence , also truncating if it reaches excess in post
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.Sequential([
    # embedding layer in Keras can be used when we want to create the embeddings to embed higher dimensional data into lower dimensional vector space
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), #The first argument is the number of distinct words in the training set. The second argument indicates the size of the embedding vectors. The input_length argumet determines the size of each input sequence.
    #Bidirectional LSTMs train two instead of one LSTMs on the input sequence. The first on the input sequence as-is and the second on a reversed copy of the input sequence. This can provide additional context to the network and result in faster and even fuller learning on the problem.
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #an activation function is a function that is added into an artificial neural network in order to help the network learn complex patterns in the data
    #The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
    tf.keras.layers.Dense(24, activation='relu'),
    #The function takes any real value as input and outputs values in the range 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 15

#have to convert all padded sequences and labels to arrays for model fitting
import numpy as np
training_padded = np.asarray(training_padded)
training_labels = np.asarray(training_labels)
testing_padded = np.asarray(testing_padded)
testing_labels = np.asarray(testing_labels)
#history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1,callbacks=[CustomCallbacks()])
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)
model.save("chatbot-nlp.h5")
'''
sentence=['Does anyone have the answer for unit 5','https://www.youtube.com/watch?v=LjKmsmHulzM']
sequences=tokenizer.texts_to_sequences(sentence)
padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
print(model.predict(padded))'''