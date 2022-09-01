import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from scipy import stats
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense, Activation
from tkinter import *

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)

print("Type: ", type(X_train))
print("Type: ", type(Y_train))

print("X train shape: ", X_train.shape)
print("Y train shape: ", Y_train.shape)







# %% EDA

print("Y train values: ", np.unique(Y_train))
print("Y test values: ", np.unique(Y_test))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y_test distribution: ", dict(zip(unique, counts)))

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

d = X_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []
for i, ii, in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii)) 

sns.distplot(review_len_train, hist_kws = {"alpha":0.3})     
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})     

print("Train mean: ", np.mean(review_len_train))
print("Train median: ", np.median(review_len_train))
print("Train mode: ", stats.mode(review_len_train))

#number of words

word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items():
    if values == 4:                             #example
        print(keys)

def whatItSay(index = 23):
    
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatItSay(3)

#%%   Preprocessing

num_words = 15000   # 88584 tane unique kelime olduğu için 15000'e indirgedik.
(X_train,Y_train), (X_test, Y_test) = imdb.load_data(num_words = num_words)

maxlen = 130
X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen = maxlen)

print(X_train[5])

for i in X_train[0:10]:
    print(len(i))
    
#decoded_review = whatItSay(5)

#%% RNN

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=5, batch_size=128, verbose=1)

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %",score[1]*100)

#Accuracy Plot
plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

#Loss Plot
plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

#%% GUI

def good_comment():
    for i in range(0,24999): 
        random_number = random.randint(0,24999)
        if(Y_train[random_number]==1):
            text_box.delete(1.0, 'end')
            text_box.insert('1.0', whatItSay(random_number))
            break

def bad_comment():
    for i in range(0,24999): 
        random_number = random.randint(0,24999)
        if(Y_train[random_number]==0):
            text_box.delete(1.0, 'end')
            text_box.insert('1.0', whatItSay(random_number))
            break

ws = Tk()
ws.title('IMDB Sentimental Analysis')
ws.geometry('900x600')
ws.config(bg='#124272')

text_box = Text(
    ws,
    height=20,
    width=105
)

text_box.pack(expand=True)

#text_box.insert('end', message)

Button(
    ws,
    text='Good Comment',
    width=15,
    height=2,
    command=good_comment
).pack(expand=True)

Button(
    ws,
    text='Bad Comment',
    width=15,
    height=2,
    command=bad_comment
).pack(expand=True)

ws.mainloop()








