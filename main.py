# Import necessary modules.
import json
import string
import random

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Open and read JSON file.
with open(r"C:\Users\danny\OneDrive\Documents\chatbot\intents.json") as file:
    data = file.read()
    data = json.loads(data)

# print(data)

# Extracting words and tags.
words = []  # bag of words model for patterns.
classes = []  # bag of words model for tags.
data_x = []  # tokenized patterns.
data_y = []  # tokenized tags.

# Iterating over all intents.
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent['tag'])

        # Adding missing tags to the classes list.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Initialize lemmatizer to get stem of words.
lemmatizer = WordNetLemmatizer()

# lemmatize and convert words list and convert them to lowercase if  not punctuations.
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Sort and remove duplicates in words and classes lists.
words = sorted(set(words))
classes = sorted(set(classes))

# CONVERTING TEXT TO NUMBERS USING BAG OF WORDS MODEL.
training = []
output_empty = [0] * len(classes)

# Creating the Bag of Words Model.
for index, document in enumerate(data_x):
    bag_of_words = []
    text = lemmatizer.lemmatize(document.lower())
    for word in words:
        bag_of_words.append(1) if word in text else bag_of_words.append(0)

    # mark the index of class the current pattern is associated to.
    output_row = list(output_empty)
    output_row[classes.index(data_y[index])] = 1

    # add the one hot encoded bag of words and associated classes to training.
    training.append([bag_of_words, output_row])

# Shuffle the data and convert it to an array.
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the lemmatized patterns and tags.
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# CREATE AND TRAIN THE NEURAL NETWORK.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

# print(model.summary())
model.fit(x=train_x, y=train_y, epochs=200, verbose=1)


# PREPROCESSING USER INPUT.
def clean_text(text):
    """Receives text as string, strips it to its root form
    using lemmatizer and returns that.
"""
    input_tokens = nltk.word_tokenize(text)
    input_tokens = [lemmatizer.lemmatize(word.lower()) for word in input_tokens]
    return input_tokens


def bag_of_words(text, vocabulary):
    """Calls the above function and converts the text to array using
    the bag of words model using the input vocabulary argument and
    returns the same array."""
    tokens = clean_text(text)
    bag_of_words = [0] * len(vocabulary)
    for token in tokens:
        for index, word in enumerate(vocabulary):
            if word == token:
                bag_of_words[index] = 1
    return np.array(bag_of_words)


def predict_class(text, vocabulary, labels):
    """Returns a list that contains a tag corresponding to the highest
    probability."""
    bow = bag_of_words(text, vocabulary)
    result = model.predict(np.array([bow]))[0]  # Extracting probabilities.
    thresh = 0.5
    y_predict = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_predict.sort(key=lambda x: x[1], reverse=True)  # Sort by probability values in reverse order.
    return_list = []
    for item in y_predict:
        return_list.append(labels[item[0]])  # Contains tags for highest probability.
    return return_list


def get_response(intents_list, intents_json):
    """Takes a tag returned from the prior function and randomly uses it to
    select a response corresponding to the same tag in intents.json file.
    The return list is empty when the probability does not pass the threshold
     as such, 'Sorry! I do not understand' becomes the chatbot's response."""
    if len(intents_list) == 0:
        result: str = 'Sorry! I do not understand.'
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for item in list_of_intents:
            if item["tag"] == tag:
                result = random.choice(item["responses"])
                break
    return result


# INTERACTING WITH THE CHATBOT.
print(F"Press 0 to exit the chatbot.")
while True:
    message = input("")
    if message == "0":
        break
    intents = predict_class(text=message, vocabulary=words, labels=classes)
    result = get_response(intents_list=intents, intents_json=data)
    print(result)
