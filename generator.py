import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras as k
import random as rnd

word_to_id_map = None
id_to_word = None
vocabulary_size = None
none_id = None

def word_to_id(word):
    if word is None:
        word = "_"
    word = word.lower()
    if word not in word_to_id_map.keys():
        word = "*"
    return word_to_id_map[word]

def load_vocabulary():

    global word_to_id_map
    global id_to_word
    global vocabulary_size
    global none_id

    print()
    print("Loading words...", end=" ")

    with open("./data/vocabulary.txt", "r", encoding="utf-8") as vocabulary_file:
        words = vocabulary_file.readlines()

    words = [word[:-1] for word in words]

    word_to_id_map = {word: id for id, word in enumerate(words)}
    id_to_word = {id: word for id, word in enumerate(words)}

    vocabulary = word_to_id_map.keys()
    vocabulary_size = len(vocabulary)
    none_id = word_to_id_map["_"]

    print("Done")

model = None
context_size = None

AVAILABLE_CONTEXT_SIZES = [1, 2, 3]

def read_context_size():

    global context_size

    print()
    print(f"Available context sizes: {AVAILABLE_CONTEXT_SIZES}.")
    print("The context size is the number of words previously generated the model will take into account.")
    context_size = int(input("Insert the context size: "))

def load_model():
    
    global model
    global context_size

    print()
    print("Loading model...", end=" ")

    input_layer = k.layers.Input(
        name="input",
        shape=(vocabulary_size * context_size,)
    )

    hidden_layer = k.layers.Dense(
        name="hidden",
        units=(vocabulary_size * context_size + vocabulary_size) // 2,
        activation="sigmoid"
    )(input_layer)

    output_layer = k.layers.Dense(
        name="output",
        units=vocabulary_size,
        activation="softmax"
    )(hidden_layer)

    model = k.models.Model(
        name=f"model-{context_size}",
        inputs=input_layer,
        outputs=output_layer
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics="categorical_accuracy"
    )

    model.load_weights(f"./model/weights-{context_size}.ckpt").expect_partial()

    print("Done")

context = None
sentence = None

def initialize():

    global context
    global sentence

    context = [none_id] * context_size
    sentence = ""

def show_how_to():
    print()
    print("HOW TO USE:")
    print()
    print("With your guidence, this program will create a sentence. The sentence will be gradually formed and")
    print("shown in the screen. At each iteration, you will be asked to insert a command from this list:")
    print()
    print("> gen")
    print("    Generates the next word.")
    print()
    print("> gen [k]")
    print("    Generates the next k words. Example: gen 15")
    print()
    print("> add [s]")
    print("    Adds the string s to the sentence. Example: add going to the supermarket")
    print()
    print("> help")
    print("    Shows this guide.")
    print()
    print("> stop")
    print("    Ends the sentence and the program.")
    print()
    print("If you provide an unknown command or none at all, the program will simply generate the next word.")
    print("Provide arguments in the expected format. This program does not validate input.")
    print()
    input("Press [Enter] to resume. ")
    print()

def make_prediction():

    global context

    model_input = tf.zeros(context_size * vocabulary_size)
    for i, word_id in enumerate(context):
        final_index = word_id + vocabulary_size*i
        model_input += tf.one_hot(final_index, depth=vocabulary_size*context_size)
    model_input = tf.reshape(model_input, (1, -1))

    prediction = model(model_input)
    return prediction.numpy()[0]

def choose_word_id(prediction):
    return rnd.choices([i for i in range(vocabulary_size)], weights=prediction)[0]

def add_last_word(word, word_id=None):

    global context
    global sentence

    if word_id == None:
        word_id = word_to_id(word)
    
    if context[-1] == none_id:
        word = word.capitalize()
    
    sentence = f"{sentence} {word}"
    context.append(word_id)
    context = context[1:]

def generate_last_word():
    prediction = make_prediction()
    word_id = choose_word_id(prediction)
    word = id_to_word[word_id]
    add_last_word(word, word_id)

def generate_last_words(count):
    for i in range(count):
        generate_last_word()

def run_generator():

    global sentence

    while True:
        
        print(f"Sentence: {sentence}")

        arguments = input("> ").split(" ")
        command = arguments[0]

        if command == "stop":
            break

        elif command == "gen":
            if len(arguments) == 1:
                generate_last_word()
            else:
                count = int(arguments[1])
                generate_last_words(count)
            continue

        elif command == "add":
            words = arguments[1:]
            for word in words:
                add_last_word(word)
            continue

        elif command == "help":
            show_how_to()
            continue

        generate_last_word()

    sentence = f"{sentence}."
    print(f"Sentence: {sentence}")

if __name__ == "__main__":
    load_vocabulary()
    read_context_size()
    load_model()
    initialize()
    show_how_to()
    run_generator()