# our rnn.py

# implementation of an RNN that takes and trains on note sequences

import magenta
import note_seq
from note_seq import chord_inference
import magenta.scripts.convert_dir_to_note_sequences as scripts
import tensorflow.compat.v1 as tf
from tensorflow.data import Dataset, TFRecordDataset
from note_seq import midi_io
import os
from note_seq import sequences_lib # https://github.com/magenta/note-seq/blob/a7ea6b3ce073b0791fc5e89260eae8e621a3ba0c/note_seq/chord_inference.py for quantization of note_seq
from note_seq import chords_encoder_decoder
import numpy as np
import pandas as pd

SEQ_LEN = 25
NUM_MIDI_VALS = 128

# define batch size
BATCH_SIZE = 64

def prepreocessing():
    tf.logging.set_verbosity(tf.logging.INFO)
    main_github_dir = os.getcwd() # get cwd
    input_dir = main_github_dir + '/PianoCompetitionMidi/' # get midi folder

    # must output to TFRecord file otherwise note sequence is not able to be parsed
    #output_dir = main_github_dir + '/tmp/notesequences.tfrecord'

    # walk through files and create one hot encoding of each note sequence in large array
    onehot_arr = []

    # define a dictionary that holds file names as keys and their note sequence encodings as values
    file_name_to_seq_encoding = dict()


    chord_dict = dict()

    for (root,dirs,files) in os.walk(input_dir, topdown=True):
        # need all root, dirs, files in order to get the files alone
        # loop though all the files
        filename_chord_list = []
        all_notes = []
        for filename in files:
            #print(filename)
            if len(dirs) == 0:
                filepath = root + '/' + filename
                
                # convert midi file to note seq
                note_sequence = midi_io.midi_file_to_note_sequence(filepath)
                #print(note_sequence.notes[0].pitch)
                notes = extract_notes(note_sequence)
                """ for key in notes:
                    if key == 'pitch':
                        #print(notes[key])
                print('----------------------') """

                # getting large collection of all the notes in all files
                all_notes.append(notes)

                n_notes = len(all_notes)

                sequence = format_data(all_notes)

                shorter_seq = sequence[:SEQ_LEN]

                # Format dataset further by creating batches
                # batches allow us to pass in multiple instances of the training set at one, faster overall

                num_dataset_items = n_notes - SEQ_LEN # number of items in dataset

                # shuffle the dataset to be random, and create batches


                #print(shorter_seq)

                # Create model, copied from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=kNaVWcCzAm5V
                # TODO: below is all copy pasted

                input_shape = (SEQ_LEN, 3) # gets shape / dimensions of input
                #print(input_shape)
                learning_rate = 0.005

                inputs = tf.keras.Input(input_shape)
                print(inputs)
                x = tf.keras.layers.LSTM(128)(inputs) # defines LSTM layer with 128 things for # of midi values

                # define output layer of dense layers
                outputs = {
                    # define hidden layers, specifying amount of possible values of the output
                    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
                    'step': tf.keras.layers.Dense(1, name='step')(x),
                    'duration': tf.keras.layers.Dense(1, name='duration')(x),
                }

                model = tf.keras.Model(inputs, outputs) # define model

                loss = {
                    # Computes the crossentropy loss between the labels and predictions

                    # logits the vector of raw (non-normalized) predictions that a classification model generates,
                    # Probability of 0.5 corresponds to a logit of 0. Negative logit correspond to probabilities less than 0.5, positive to > 0.5.


                    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True),
                    'step': mse_with_positive_pressure,
                    'duration': mse_with_positive_pressure,
                }

                # optimizer implements Adam algorithm which is a stochastic gradient descent algorithm with smoother error correction
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                model.compile(loss=loss, optimizer=optimizer)

                model.summary()
                
                
                
                    
                

                

    #print(file_name_to_seq_encoding) 

    for key in file_name_to_seq_encoding:
        print(key, " : ")
        print(file_name_to_seq_encoding[key])
        print("------------------")


    
def chord_formatting(arr):
    is_chord = arr[0] # if 0 is chord, if 1 is not chord
    # skip first row and loop from 13th item till end
    res = []
    
    #notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_members = arr[13: 25] # gets middle row of chord_members
    
    # iterate through remaining arr and get index of bass note
    index_of_bass_note = arr[25:].index(1.0)

    chord_members[index_of_bass_note] = 1.0

    res.append(is_chord)
    res.extend(chord_members)

    return res


# referenced https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb
def extract_notes(note_sequence):
    notes = dict()
    #print(note_sequence.notes[0].pitch)
    # midi pitch

    # sorted note_sequences by start_time
    sorted_note_seq = sorted(note_sequence.notes, key=lambda x: x.start_time)
    #print(sorted_note_seq)
    
    # set prev_start to be first value
    prev_start = sorted_note_seq[0].start_time

    # loop through the notes in note_sequence
    for note in sorted_note_seq:
        note_name = note.pitch
        # start time
        start = note.start_time
        end = note.end_time

        duration = end - start

        step = start - prev_start

        prev_start = start

        add_to_dict(notes, 'pitch', note_name)
        add_to_dict(notes, 'start', start)
        add_to_dict(notes, 'end', end)
        add_to_dict(notes, 'step', step)
        add_to_dict(notes, 'duration', duration)
    
    # return notes dictionary (as pd frame)
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()}) # copied

def add_to_dict( dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


# return a sequence of a dataset of tensors
def format_data(all_notes):
    # iterate through np array of dataframes
    res = []
    for i in range(len(all_notes)):
        current = all_notes[i]
        print(type(current))
        # np_pitch_step_duration = np.array()
        np_data = current[["pitch","step","duration"]].to_numpy()
        
        # made np_data into tensor
        tensor = tf.constant(np_data, dtype=tf.float64)
        res.append(tensor)
        #print("teeeensor: ", tensor)
    return res

# separate sequences into labels and inputs
# sequences represent all the note sequences
# a single sequence is 1 note sequence
def separate_labels(sequences): # need to put labels within the sequences list
    inputs = sequences[1:]
    labels = sequences[0]
    return inputs, labels


#NOTE: Copied from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=erxLOif08e8v
#Generates Mean Squared Error like normal but includes this idea of positive pressure.
#Positive pressure a method to increase the error value which then forces the network
#to make larger adjustments to the weights, to push the output values back up to a positive value.
#If the predicted value is already positive, the positive pressure will be 0 and have no influence on the error.
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)


prepreocessing()
