from helper import preprocess
import pandas as pd
from Models import model
import np_utils
import numpy as np
import random
import tflearn as tfl
"""
" remove via replace!
"""

data=open("./Datasets/short-jokes/shortjokes.csv","r").read().replace('"',"")
#print(pd.DataFrame(data))
p=preprocess.preprocess()
m=model.RNN(model_path="A:/Github/JokeAi/Models/model2.tfl")
#p.word_setup("./Datasets/short-jokes/shortjokes.csv","ID")
#data=open("./new_csv.csv","r").read()
chars=sorted(list(set(data)))
int_to_w={}
w_to_int={}

for i,n in enumerate(chars):
    int_to_w[i]=n
    w_to_int[n]=i

text_len=50

X=[]
Y=[]
#i=1080000
for i in range(0,len(data),1):
    print("{0} von {1}".format(i,len(data)))
    x=data[i:i+text_len]#first 50 letters
    y=data[i+text_len]#the i th +50th letter
    X.append([w_to_int[q] for q in x])
    Y.append(w_to_int[y])
    if i==10800000:
        break
x=np.reshape(X,(len(X),text_len,1))
print(x.shape)
y=tfl.data_utils.to_categorical(Y,len(chars))
#m.cont_train(x,y)
#x=x/float(len(chars))
m.train(x,y)
"""
m=model.RNN()
test=[]
sentence="Man "
for i in range(100):
    test=[]
    for i in sentence:
        test.append(w_to_int[i])
    test=np.reshape(test,(1,len(test),1))
    #test=test/len(chars)
    #q=model.predict(model)
    q=m.predict(test)
    sentence=sentence+int_to_w[q]
    print("\n")
    print(sentence)
    print("\n")
"""
