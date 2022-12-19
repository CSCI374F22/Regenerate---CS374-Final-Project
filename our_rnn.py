# our rnn.py

# implementation of an RNN that takes and trains on note sequences

import magenta
import note_seq
import os
import numpy as np
import pandas as pd
import magenta.scripts.convert_dir_to_note_sequences as scripts
import tensorflow.compat.v1 as tf
import random
import keyfindingalg
import math
import mido


from note_seq import sequences_lib
from note_seq import chords_encoder_decoder
from note_seq import chord_inference
from tensorflow.data import Dataset, TFRecordDataset
from note_seq import midi_io
from note_seq import sequences_lib # https://github.com/magenta/note-seq/blob/a7ea6b3ce073b0791fc5e89260eae8e621a3ba0c/note_seq/chord_inference.py for quantization of note_seq
from note_seq import chords_encoder_decoder
from magenta.pipelines import note_sequence_pipelines
from magenta.common import testing_lib as common_testing_lib
from note_seq.protobuf import music_pb2
from magenta.pipelines import statistics
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation

SEQ_LEN = 32
NUM_MIDI_VALS = 128

# define batch size
BATCH_SIZE = 64

# define universal key
MASTER_MAJOR_KEY = 'C' # C major
MASTER_MINOR_KEY = 'A' # A minor

SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def prepreocessing():
    tf.logging.set_verbosity(tf.logging.INFO)
    main_github_dir = os.getcwd() # get cwd
    input_dir = main_github_dir + '/Piano_MIDI_Files/Piano_E_Competition_2011/Piano_E_Competition_2011_1/' # get midi folder

    # must output to TFRecord file otherwise note sequence is not able to be parsed
    #output_dir = main_github_dir + '/tmp/notesequences.tfrecord'

    all_note_sequences = []

    for (root,dirs,files) in os.walk(input_dir, topdown=True):
        # need all root, dirs, files in order to get the files alone
        # loop though all the files
        for filename in files:

            if len(dirs) == 0:
                filepath = root + '/' + filename
                
                # convert midi file to note seq
                note_sequence = midi_io.midi_file_to_note_sequence(filepath)


                # get key of filename

                # get key of filename
                key, keyinfo = keyfindingalg.get_key(str(filepath))
                letter_list = key[0].split()
                letter = letter_list[0]

                if (keyinfo == 'major'):
                    master_index = SCALE.index(MASTER_MAJOR_KEY)
                else:
                    master_index = SCALE.index(MASTER_MINOR_KEY)
               
                current_index = SCALE.index(letter)
                transpose_int = master_index - current_index
                
                # transpose to universal key
                if (transpose_int == 0): # means file is already in master major or minor key
                    #print("Gurrrrl, you got da master key")
                    continue

                # transpose 

                transposed = transpose(note_sequence, transpose_int)

                note_seq.sequence_proto_to_midi_file(transposed, 'transpositions/transposed-' + filename)

                seq = extract_notes(transposed)
                all_note_sequences.append(seq)
    
    # transpose it and change its shape to be the total of all the rows of the note_sequences
    all_note_sequences = pd.concat(all_note_sequences)

    # Format dataset further by creating batches
    # batches allow us to pass in multiple instances of the training set at one, faster overall

    # shuffle the dataset to be random, and create batches

    random.shuffle(list(all_note_sequences)) # shuffle random # put back in

    # needs to include the last notes (pitch, step, duration) of ALL the note sequences, not just one
    labels = []
    for i in range(len(all_note_sequences)):
        pitches = all_note_sequences["pitch"].iloc[i]
        steps = all_note_sequences["step"].iloc[i]
        duration = steps = all_note_sequences["duration"].iloc[i]
        final = np.array([pitches,steps, duration]).T
        labels.append(final[-1])

    inputShape = (SEQ_LEN, 3) # gets shape / dimensions of input

    # define model
    model = Sequential()
    model.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model.add(Dense(108))
    model.add(Dense(1))
    model.add(Dense(1))

    # define model2
    model2 = Sequential()
    model2.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model2.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model2.add(Dense(1))
    model2.add(Dense(1))
    model2.add(Dense(1))

    # define model3
    model3 = Sequential()
    model3.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model3.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model3.add(Dense(1))
    model3.add(Dense(1))
    model3.add(Dense(1))

    # optimizer implements Adam algorithm which is a stochastic gradient descent algorithm with smoother error correction
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model2.compile(optimizer='adam', loss='mse')
    model3.compile(optimizer='adam', loss='mse')

    model.build(inputShape)

    model.summary()
    model2.summary()
    model3.summary()

    # Copied callbacks from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=uQA_rwKEgPjp

    # callback is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved

    # specific callbacks for pitch
    callbacks_pitch = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_pitch_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    # prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    ]

    epochs = 50

    # specific callbacks for step
    callbacks_step = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_step_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    # prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    ]

    # specific callbacks for duration
    callbacks_duration = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_pitch_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    # prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    ]

    epochs = 50

    # model.evaluate
    for x_batch, y_batch in get_batch(all_note_sequences[0:1000], labels): # must be long enough, at least 100
        x_batch = x_batch.to_numpy()
        resized_batch = x_batch.reshape(-1,SEQ_LEN,3)
        y_batch = np.asarray(y_batch)
        resized_y = y_batch.reshape(-1,SEQ_LEN)
        model.evaluate(resized_batch, resized_y, return_dict=True)
        model2.evaluate(resized_batch, resized_y, return_dict=True)
        model3.evaluate(resized_batch, resized_y, return_dict=True)

    # referenced for formatting for loop https://datascience.stackexchange.com/questions/108099/is-fitting-a-model-in-a-for-loop-equivalent-to-using-epochs1
    

    #for epoch in range(epochs):
    # goes through all of the batches (training data and associated labels) in all_note_sequence[0:1000]
    for x_batch, y_batch in get_batch(all_note_sequences[0:1000], labels):
        num_steps_per_epoch = len(x_batch) // BATCH_SIZE
        x_batch = x_batch.to_numpy()
        resized_batch = x_batch.reshape(-1,SEQ_LEN,3)
        #print("x batch: ", resized_batch)
        y_batch = np.asarray(y_batch)
        resized_y = y_batch.reshape(-1,SEQ_LEN)
        
        model.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_pitch, epochs=epochs)
        model2.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_step, epochs=epochs)
        model3.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_duration, epochs=epochs)


    # generate notes
            
    num_generated_notes = 1000 # num of notes to generate

    prev_start = 0 # set prev_start

    res = music_pb2.NoteSequence() # create new note sequence


    key_order = ['pitch', 'step', 'duration']
    
    # getting 32 of the all_note_sequences inputs
    input_notes = np.stack([all_note_sequences[key][:32] for key in key_order], axis=1)


    for i in range(num_generated_notes):
        
        # get generated pitch, step, duration
        pitch = predict_pitch(model, input_notes)
        step = predict_step(model2, input_notes)
        duration = predict_duration(model3, input_notes)

        
        start = prev_start + step
        end = start + duration

        res.notes.add(pitch=pitch, start_time=start, end_time=end, velocity=80)

        prev_start = start

    res.tempos.add(qpm=60)

    note_seq.sequence_proto_to_midi_file(res, 'generated_piece.mid')

    generated_midi = mido.MidiFile('generated_piece.mid')

    print('🎉 Done!', generated_midi)


# ended up not being used
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


# somewhat copied https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb
# extracts individual notes from a note sequence, specifically getting 'pitch', 'duration', and 'step' and returns a dataframe
def extract_notes(note_sequence):
    notes = dict()

    # sorted note_sequences by start_time
    sorted_note_seq = sorted(note_sequence.notes, key=lambda x: x.start_time)
    
    # set prev_start to be first value
    prev_start = sorted_note_seq[0].start_time

    # loop through the notes in note_sequence
    for note in sorted_note_seq:
        note_name = note.pitch
        
        # start time
        start = note.start_time
        end = note.end_time

        # duration
        duration = end - start

        step = start - prev_start

        prev_start = start

        add_to_dict(notes, 'pitch', note_name)
        add_to_dict(notes, 'step', step)
        add_to_dict(notes, 'duration', duration)
    
    # return notes dictionary (as pd frame)
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()}) # copied

# adds key, value pairs to dictionary
def add_to_dict( dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


# didn't end up using this function
# return a sequence of a dataset of tensors
def format_data(all_notes):
    # iterate through np array of dataframes
    array_list = []
    for i in range(len(all_notes)):

        current = all_notes.iloc[i] # single not sequence, of type series
        np_data = current[["pitch", "step", "duration"]].to_numpy()

        final = np.array(np_data)
        array_list.append(final)
    
    ns_dataset = tf.data.Dataset.from_tensor_slices(array_list)
    
    return ns_dataset

# didn't end up using this function
# separate sequences into labels and inputs
# sequences represent all the note sequences
# a single sequence is 1 note sequence
def separate_labels(sequences): # need to put labels within the sequences list
    length = len(sequences)
    inputs = sequences[:length-1] 
    labels = sequences[length-1]
    return inputs, labels


# copied from https://stackoverflow.com/questions/50539342/getting-batches-in-tensorflow since we didn't know how else to get batches
# get batch gets batches in the form of a generator object, which when iterated through gives all the batches
# for a specific sequence
def get_batch(inputX, label):
    length = len(inputX)

    # split data passed in into batches
    for i in range(0, (length//BATCH_SIZE)):

        # getting index of each batch
        index = i * BATCH_SIZE
        # getting each batch and its corresponding label
        # yield: sends value back to caller but maintains enough state to resume where function left off
        # yield returns a generator object
        yield inputX[index: index + BATCH_SIZE], label[index: index + BATCH_SIZE]


# transpose takes in a note sequence and adds an amount to each note in the sequence to transpose it to a diff key, and return transposed note sequence
def transpose(note_sequence, amount):
    res = music_pb2.NoteSequence()

    sorted_note_seq = sorted(note_sequence.notes, key=lambda x: x.start_time)
    
    # set prev_start to be first value
    prev_start = sorted_note_seq[0].start_time
    
    for note in sorted_note_seq:

        # tranpose note by amount
        new_note = note.pitch + amount

        res.notes.add(pitch=new_note, start_time=note.start_time, end_time=note.end_time, velocity=note.velocity)

    res.total_time = note_sequence.total_time
    
    return res

# referenced https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=X0kPjLBlcnY6
# predict_pitch takes in a model and notesequences, running model.predict on the reshaped notesequences
# which are reshaped in order to match the output of model.fit
# function outputs a single pitch from the predicted_pitches as an int
def predict_pitch(model, notesequences):
    # determines randomness
    temperature = 3
    reshaped_inputs = tf.expand_dims(notesequences, 0)

    predictions = model.predict(reshaped_inputs)
    predictions /= temperature

    # np.argmax returns the indices of the max values along an axis
    pitch = np.argmax(predictions, axis = 1)
    pitch = tf.squeeze(pitch, axis=-1)

    return int(pitch)


# referenced https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=X0kPjLBlcnY6
# predict_step takes in a model and notesequences, running model.predict on the reshaped notesequences
# which are reshaped in order to match the output of model.fit
# function outputs a single step from the predicted_steps as an int
def predict_step(model, notesequences):

    reshaped_inputs = tf.expand_dims(notesequences, 0)

    predictions = model.predict(reshaped_inputs)

    step = np.argmax(predictions, axis = 1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)

    return int(step)

# referenced https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=X0kPjLBlcnY6
# predict_duration takes in a model and notesequences, running model.predict on the reshaped notesequences
# which are reshaped in order to match the output of model.fit
# function outputs a single duration from the predicted_durations as an int
def predict_duration(model, notesequences):

    reshaped_inputs = tf.expand_dims(notesequences, 0)

    predictions = model.predict(reshaped_inputs)

    duration = np.argmax(predictions, axis = 1)
    duration = tf.squeeze(duration, axis=-1)

    # `step` and `duration` values should be non-negative
    duration = tf.maximum(0, duration)

    return int(duration)







prepreocessing()
