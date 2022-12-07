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
        for filename in files:
            #print(filename)
            if len(dirs) == 0:
                filepath = root + '/' + filename
                
                # convert midi file to note seq
                note_sequence = midi_io.midi_file_to_note_sequence(filepath)


                # trim note sequence
                subsequence = sequences_lib.trim_note_sequence(note_sequence, 0, 60)

                # referenced https://github.com/magenta/note-seq/blob/a7ea6b3ce073b0791fc5e89260eae8e621a3ba0c/note_seq/sequences_lib.py#L988
                """Quantize a NoteSequence proto relative to tempo.
                The input NoteSequence is copied and quantization-related fields are
                populated. Sets the `steps_per_quarter` field in the `quantization_info`
                message in the NoteSequence.
                Note start and end times, and chord times are snapped to a nearby quantized
                step, and the resulting times are stored in a separate field (e.g.,
                quantized_start_step). See the comments above `QUANTIZE_CUTOFF` for details on
                how the quantizing algorithm works.
                Args:
                    note_sequence: A music_pb2.NoteSequence protocol buffer.
                    steps_per_quarter: Each quarter note of music will be divided into this many
                    quantized time steps."""

                # quantize note sequence
                quantized_sequence = sequences_lib.quantize_note_sequence(
                subsequence, steps_per_quarter=4)
                # infer chords
                chord_inference.infer_chords_for_sequence(
                    quantized_sequence)
                #print(note_sequence)

                # get all chords in quantized ntoe sequence
                chords = [(ta.text, ta.time) for ta in quantized_sequence.text_annotations] 
                #for ta in quantized_sequence.text_annotations:    
                    #print(ta.text, ta.time)

                # get encodings of all chords in note sequence
                arr = []

                # define encoder
                enc = chords_encoder_decoder.PitchChordsEncoderDecoder()

                for chord in chords:
                    # get chord name
                    chord_name = chord[0]
                    # encoding array

                    # depending on how we're using the one hot encoding it might be better to not use this
                    # way since it makes things really long
                    encoding = enc.events_to_input([chord_name], 0)
                    formatted_encoding = chord_formatting(encoding)
                    #print(formatted_encoding)
                    arr.append(formatted_encoding)
                    #print("array now: ", arr)

                    chord_dict[chord_name] = formatted_encoding
                    filename_chord_list.append(chord_name)
                

                file_name_to_seq_encoding[filepath] = filename_chord_list

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




prepreocessing()

