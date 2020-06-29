# importing the libraries
from tkinter import *
import random
import json
from tensorflow.python.keras.models import load_model
import re
import numpy as np
import pickle
import nltk
from keras import Input
from keras.layers import LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from nltk.stem import WordNetLemmatizer
from keras.models import Model

# making lemmatizer object
lemmatizer = WordNetLemmatizer()
# importing keras model

model = load_model('chatbot_model.h5')

# importing json

intents = json.loads(open('intents.json').read())
context = ""
context1 = ""
padding_type = 'post'
trunc_type = 'post'

with open('tokenizer.json') as t:
    data = json.load(t)
    tokenizer = tokenizer_from_json(data)
with open('label_tokenizer.json') as t:
    data = json.load(t)
    label_tokenizer = tokenizer_from_json(data)


# data file
data_file = open('intents.json').read()
intents = json.loads(data_file)
Questions = []
labels = ['greeting', 'goodbye', 'thanks', 'options', 'adverse_drug',
          'blood_pressure', 'blood_pressure_search', 'pharmacy_search', 'hospital_search']
Answers = []


# function to filter predictions lower than a threshold
def predict_class(sentence):
    sentence = [sentence]
    p = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(
        p, maxlen=10, padding=padding_type, truncating=trunc_type)
    res = model.predict(padded)
    result = np.argmax(res)
    result1 = labels[result-1]
    return result1


# function to get response
def getResponse(intstag, intents_json):
    tag = intstag
    listOfIntents = intents_json['intents']
    for i in listOfIntents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            global context1
            context1 = i['context'][0]
            break
    return result

# function to get chatbot response


def chatbot_response(msg):
    intstag = predict_class(msg)
    print("Context " + intstag)
    if (context1 != ""):
        intstag = context1
    res = getResponse(intstag.strip(), intents)
    return res


# creating GUI with Tkinter


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#635a6c", font=("Monospace", 12))
        global context
        if(context == "goodbye"):
            with open("Chatlog.txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write("User: " + msg + "\n")
                file_object.write("\n\n\n")
                ChatLog.insert(
                    END, "Bot: " + "Thank you for this rating, it's valuable to us" + '\n\n')
            context = ""

        else:
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            with open("Chatlog.txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write("User: " + msg + "\n")
                file_object.write("Bot: " + res)
                context = predict_class(msg)
                if (context == "goodbye"):
                    if (context1 == ""):
                        ChatLog.insert(
                            END, "Bot: Please provide a rating for this chat out of 5\n\n")
                        file_object.write(
                            "\nBot: Please provide a rating for this chat out of 5")
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello there!")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)


# Creating chat window
ChatLog = Text(base, bd=0, bg="white", height="8",
               width="50", font="Monospace", )

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#dd6a31", activebackground="#f94f06", fg='#dedede',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29",
                height="5", font="Monospace")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=281, y=401, height=90, width=110)

base.mainloop()
