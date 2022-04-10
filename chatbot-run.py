from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras import models
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = models.load_model('C:/Users/Tarun/Desktop/comp/pycharm/NLP/chatbot-nlp.h5')

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

#Tokenization is a method to segregate a particular text into small chunks or tokens. Here the tokens or chunks can be anything from words to characters, even subwords.
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) #num_words -> num of tokens , Out Of Vocab token - this will replace any unknown words with a token of our choosing
tokenizer.fit_on_texts(training_sentences) #fits tokenizer on the sentences


sentence=['Does anyone have the answer for unit 5','link to join my WhatsApp group','can someone send the question paper']
sequences=tokenizer.texts_to_sequences(sentence)
padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
pred=model.predict(padded)
#print(pred)
count=0
for i in pred:
    if(i>=0.6):
        print(i,sentence[count]," : irrelevant information")
        count+=1
    else:
        print(i, sentence[count], " : relevant information")
        count += 1