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
    input_dir = main_github_dir + '/PianoCompetitionMidi/' # get midi folder

    # must output to TFRecord file otherwise note sequence is not able to be parsed
    #output_dir = main_github_dir + '/tmp/notesequences.tfrecord'

    # walk through files and create one hot encoding of each note sequence in large array
    onehot_arr = []

    # define a dictionary that holds file names as keys and their note sequence encodings as values
    file_name_to_seq_encoding = dict()


    chord_dict = dict()
    all_note_sequences = []
    count = 0
    for (root,dirs,files) in os.walk(input_dir, topdown=True):
        # need all root, dirs, files in order to get the files alone
        # loop though all the files
        filename_chord_list = [] 
        for filename in files:
            #print(filename)
            if len(dirs) == 0:
                filepath = root + '/' + filename
                
                # convert midi file to note seq
                note_sequence = midi_io.midi_file_to_note_sequence(filepath)


                # get key of filename

                # get key of filename
                key, keyinfo = keyfindingalg.get_key(str(filepath))
                letter = key[0][0]
                #print(letter)
                if (keyinfo == 'major'):
                    master_index = SCALE.index(MASTER_MAJOR_KEY)
                else:
                    master_index = SCALE.index(MASTER_MINOR_KEY)
               
                current_index = SCALE.index(letter)
                transpose_int = master_index - current_index

                #print("curr key: ", key, "master: ", master_index)

                dataframe_of_notes = extract_notes(note_sequence)
                #pitches = list(dataframe_of_notes['pitch'])

                # get min and max pitches
                #min_pitch = min(pitches)
                #max_pitch = max(pitches)
                """ print("curr key: ", letter, ", transposed key: ", SCALE[master_index])
                print(filename + ": transpose int: ", transpose_int) """
                #print(filename, key, keyinfo)
                
                

                    

                # transpose to universal key
                if (transpose_int == 0): # means file is already in master major or minor key
                    #print("Gurrrrl, you got da master key")
                    continue

                # transpose 

                transposed = transpose(note_sequence, transpose_int)

                note_seq.sequence_proto_to_midi_file(transposed, 'transpositions/transposed-' + filename)

                #print(typ\e(transposed))

                #key, keyinfo = keyfindingalg.get_key(str(filepath))
                #letter = key[0][0]
                #print(letter)

                
                """ for key in notes:
                    if key == 'pitch':
                        #print(notes[key])
                print('----------------------') """

                # getting large collection of all the notes in all files
                #print('--------------------')
                seq = extract_notes(transposed)
                all_note_sequences.append(seq)
                #count += 1

    print("Before shape: ", np.shape(all_note_sequences))
    #print("all_note_sequences (Before): ", all_note_sequences)
    
    # transpose it and change its shape to be the total of all the rows of the note_sequences
    #print("length of all_note: ", n_notes)
    #all_note_sequences = pd.concat(all_note_sequences)
    #all_note_sequences= format_data(all_note_sequences)
    
    all_note_sequences = pd.concat(all_note_sequences)
    print(all_note_sequences)

    
    print("After shape: ", np.shape(all_note_sequences))
    #print("all_note_sequences (After): ", all_note_sequences)

    n_notes = len(all_note_sequences)
    
    #print(all_note_sequences)
    #print("length ssakhafdhkj: ", n_notes)
    #print("count: ", count)
    print("all notes: ")
    print(np.shape(all_note_sequences))
    print()
    
    #print("shorter: ", shorter_seq)
    #labels = separate_labels(shorter_seq)[1]
    #print("lab: ", labels)
    # Format dataset further by creating batches
    # batches allow us to pass in multiple instances of the training set at one, faster overall

    #num_dataset_items = n_notes - SEQ_LEN # number of items in dataset

    # shuffle the dataset to be random, and create batches

    #print(shorter_seq)
    #random.shuffle(list(all_note_sequences)) # shuffle random
    
    #print("length of sequence: ", len(sequence))

    # needs to include the last notes (pitch, step, duration) of ALL the note sequences, not just one
    labels = []
    for i in range(len(all_note_sequences)):
        pitches = all_note_sequences["pitch"].iloc[i]
        steps = all_note_sequences["step"].iloc[i]
        duration = steps = all_note_sequences["duration"].iloc[i]
        #np_data3 = current["duration"].to_numpy()
        final = np.array([pitches,steps, duration]).T
        labels.append(final[-1])
    #batches = get_batch(all_note_sequences, labels) # gets generator object
    #labels = all_note_sequences
    #labels = all_note_sequences.iloc[-BATCH_SIZE:]
    # copied from https://stackoverflow.com/questions/50539342/getting-batches-in-tensorflow
    #print("batches: ", batches)
    #batch_x, batch_y = next(batches) # gets next item in the iterator batch

    """ print("batch: ", batch_inputs, batch_label)
    print("length: ", np.size(batch_inputs.to_numpy()))
    print("length: ", np.size(batch_label.to_numpy())) """
    #x = list(batches)
    #print(x)
    #batch_x, batch_y = next(batches)
    #print(sum(1 for x in batches))
    #print("batch: ", batch_inputs)
    #print("label: ", batch_label)
    


    # Create model, copied from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=kNaVWcCzAm5V
    # TODO: below is all copy pasted

    inputShape = (SEQ_LEN, 3) # gets shape / dimensions of input
    #print(input_shape)
    learning_rate = 0.005

    #inputs = tf.keras.Input(input_shape)
    #print(inputs)
    #x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs) # defines LSTM layer with 128 things for # of midi values
    # flatten
    #x = tf.keras.layers.Flatten()(x) 

    # define output layer of dense layers
    """  outputs = {
            # define hidden layers, specifying amount of possible values of the output
            'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
            'step': tf.keras.layers.Dense(1, name='step')(x),
            'duration': tf.keras.layers.Dense(1, name='duration')(x),
        } """

    #model = tf.keras.Model(inputs, outputs) # define model
    model = Sequential()
    model.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model.add(Dense(128))
    model.add(Dense(1))
    model.add(Dense(1))

    model2 = Sequential()
    model2.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model2.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model2.add(Dense(1))
    model2.add(Dense(1))
    model.add(Dense(1))

    model3 = Sequential()
    model3.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model3.add(LSTM(1, return_sequences=True, input_shape=inputShape))
    model3.add(Dense(1))
    model.add(Dense(1))
    model.add(Dense(1))

    """ loss = { # feedback
        # Computes the crossentropy loss between the labels and predictions

        # logits the vector of raw (non-normalized) predictions that a classification model generates,
        # Probability of 0.5 corresponds to a logit of 0. Negative logit correspond to probabilities less than 0.5, positive to > 0.5.


        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    } """

    # optimizer implements Adam algorithm which is a stochastic gradient descent algorithm with smoother error correction
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='mse')
    model2.compile(optimizer='adam', loss='mse')
    model3.compile(optimizer='adam', loss='mse')

    #model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    model2.summary()
    model3.summary()

    # Copied from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb#scrollTo=uQA_rwKEgPjp

    # callback is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved

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

   #print("shape: ")
    #print(np.shape(batch_inputs))
    #print(np.shape(batch_label))
    #print()
    #print("shape of short: ")
    #print(np.shape(shorter_seq))
    #np_batch_inputs = np.asarray(batch_inputs).astype('object')
    #np_batch_label = np.asarray(batch_label).astype('object')
    #batch_inputs = batch_label.to_numpy()
    #batch_label = batch_label.to_numpy()
    #resized_batch = batch_inputs.reshape(1,32,3) # size 128
    #resized_label = batch_label.reshape(1,32,1) # size 96 (somethin not right)
    #print("shapes: ", np.shape(resized_batch))
    #print("shapes: ", np.shape(resized_label))
    #print("label size: ", np.shape(resized_label))

    shorter_seq = all_note_sequences.iloc[:SEQ_LEN] # taking snippet of all that data
    print("all note sequences: ", all_note_sequences.iloc[0])
    print()
    print("short seq: ", shorter_seq.iloc[0])
    print(np.shape(shorter_seq))
    # model.evaluate

    for x_batch, y_batch in get_batch(all_note_sequences[0:1000], labels): # must be long enough, at least 100
        x_batch = x_batch.to_numpy()
        resized_batch = x_batch.reshape(-1,SEQ_LEN,3)
        y_batch = np.asarray(y_batch)
        resized_y = y_batch.reshape(-1,SEQ_LEN)
        #print("shape: ", resized_batch)
        #print("shape label: ", resized_y)
        model.evaluate(resized_batch, resized_y, return_dict=True)
        model2.evaluate(resized_batch, resized_y, return_dict=True)
        model3.evaluate(resized_batch, resized_y, return_dict=True)

    # if all_note_sequences doesn't have tensors and formatting is shit, then will kina work?
    
    # better, now labels are just wrong dimension and that's why keeps failing
    # referenced https://datascience.stackexchange.com/questions/108099/is-fitting-a-model-in-a-for-loop-equivalent-to-using-epochs1

    #try things with this dataset
    """ ns_dataset = format_data(shorter_seq)
    print("data: ", ns_dataset)
    print("shape of dataset: ", np.shape(ns_dataset))
    batch = ns_dataset.batch(BATCH_SIZE, drop_remainder=True)
    print("batch: ", batch) """

    # possible fix for warning: tensorflow:Your input ran out of data: https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
    

    #for epoch in range(epochs):
    for x_batch, y_batch in get_batch(all_note_sequences[0:1000], labels):
        num_steps_per_epoch = len(x_batch) // BATCH_SIZE
        x_batch = x_batch.to_numpy()
        resized_batch = x_batch.reshape(-1,SEQ_LEN,3)
        #print("x batch: ", resized_batch)
        y_batch = np.asarray(y_batch)
        resized_y = y_batch.reshape(-1,SEQ_LEN)

        
        #print("y batch: ", resized_y)
        #resized_y = y_batch.reshape(1,BATCH_SIZE)
        
        #print("x length: ", resized_batch.size)
        #print("y length: ", resized_y)
        print("shape: ", np.shape(resized_batch))
        #print("shape label: ", np.shape(resized_y))
        
        model.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_pitch, epochs=epochs)
        model2.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_step, epochs=epochs)
        model3.fit(resized_batch, resized_y, steps_per_epoch=num_steps_per_epoch,callbacks=callbacks_duration, epochs=epochs)


    # generate notes
            
    num_generated_notes = 1000 # num of notes to generate

    notes_generated = []

    prev_start = 0 # set prev_start

    res = music_pb2.NoteSequence() # create new note sequence

    #predict_note(model, all_note_sequences[:SEQ_LEN])


    key_order = ['pitch', 'step', 'duration']
    
    input_notes = np.stack([all_note_sequences[:SEQ_LEN][key] for key in key_order], axis=1)

    # normalize inputs
    """ input_notes = (
        notes[:SEQ_LEN] / np.array([128, 1, 1])
    ) """

    print("input notes: ", input_notes)
    print("shape: ", np.shape(input_notes))

    for i in range(num_generated_notes):
        
        # get generated pitch, step, duration
        #print("input notes: ", input_notes)
        pitch, step, duration = predict_note(model, input_notes)
        
        start = prev_start + step
        end = start + duration

        res.notes.add(pitch=pitch, start_time=start, end_time=end, velocity=80)

        prev_start = start

    res.tempos.add(qpm=60) # used to be 120

    note_seq.sequence_proto_to_midi_file(res, 'generated_piece.mid')

    generated_midi = mido.MidiFile('generated_piece.mid')

    print('🎉 Done!', generated_midi)
            

                

    #print(file_name_to_seq_encoding) 

    """  for key in file_name_to_seq_encoding:
        print(key, " : ")
        print(file_name_to_seq_encoding[key])
        print("------------------") """


    
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
        #add_to_dict(notes, 'start', start)
        #add_to_dict(notes, 'end', end)
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
    array_list = []
    for i in range(len(all_notes)):
        # current = all_notes.iloc[i] # single not sequence, of type series
        # np_pitch_step_duration = np.array()
        # np_data1 = current[["pitch", "step", "duration"]].to_numpy()
        # print("current data:",np_data1)
        #print("pitches: ", np_data1)
        #np_data2 = current["step"].to_numpy()
        #np_data3 = current["duration"].to_numpy()
        #final = np.array([np_data1,np_data2, np_data3]).T
        # final = np.array(np_data1)
        #print("final: ", final)
        # made np_data into tensor
        # tensor = tf.constant(final, dtype=tf.float64)

        current = all_notes.iloc[i] # single not sequence, of type series
        np_data = current[["pitch", "step", "duration"]].to_numpy()
        # print("current data:",np_data)

        final = np.array(np_data)
        array_list.append(final)
    
    ns_dataset = tf.data.Dataset.from_tensor_slices(array_list)
    print("dataset test:",type(ns_dataset))

    # print_list = list(ns_dataset.as_numpy_iterator())
    # for array in print_list:
    #     print(array)
    
    return ns_dataset

# separate sequences into labels and inputs
# sequences represent all the note sequences
# a single sequence is 1 note sequence
def separate_labels(sequences): # need to put labels within the sequences list
    length = len(sequences)
    inputs = sequences[:length-1] 
    labels = sequences[length-1]
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


# Feed batch
# referenced and copied from https://stackoverflow.com/questions/50539342/getting-batches-in-tensorflow
def get_batch(inputX, label):
    #print('aint you here')
    length = len(inputX)
    # print("length: ", length)
    # split data passed in into batches
    # print("range: ", length // BATCH_SIZE)
    for i in range(0, (length//BATCH_SIZE)):
        #print("i: ", i)
        # getting index of each batch
        index = i * BATCH_SIZE
        # getting each batch and its corresponding label
        # yield: sends value back to caller but maintains enough state to resume where function left off
        # yield returns an iterator
        #print("length of batch: ",len(inputX[index: index + BATCH_SIZE]))
        #print("length of label: ", len(label[index: index + BATCH_SIZE]))
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

def predict_note(model, notesequences):
    temperature = 1
    #inputs = tf.expand_dims(notesequences, 0)
    reshaped_inputs = tf.expand_dims(notesequences, 0)
    print("reshaped inputs: ", reshaped_inputs)
    print("reshaped shape: ", np.shape(reshaped_inputs))

    predictions = model.predict(reshaped_inputs)
    print("predictions: ", predictions)

    predictions /= temperature
    pitch = tf.random.categorical(predictions, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)

    print("predictions: ", predictions)
    print("pitch: ", pitch)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
  

    #print("return vals: ", int(abs(pitch)), float(updated_step_num), float(updated_duration_num))

    return int(pitch), float(step), float(duration)









prepreocessing()
