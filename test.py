import magenta
import sys
import os
from note_seq import midi_io
import keyfindingalg
import note_seq
from note_seq.protobuf import music_pb2


# 1) Converting Midi to note sequences

# https://github.com/magenta/magenta/blob/main/magenta/scripts/README.md
#INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>
#path = os.getcwd() # get current working directory

#INPUT_DIRECTORY=path + '/PianoCompetitionMidi'
# TFRecord file that will contain NoteSequence protocol buffers.
#SEQUENCES_TFRECORD=path + '/tmp/notesequences.tfrecord'

#SEQUENCE_EXAMPLES = path + '/tmp/polyphony_rnn/sequence_examples'

#RUNNING_DIRECTORY = path + '/tmp/polyphony_rnn/logdir/run1/'

#TRAINING_FILE_PATH = SEQUENCE_EXAMPLES + '/training_poly_tracks.tfrecord/'

#FINAL_OUTPUT_DIR = path + '/tmp/polyphony_rnn/'

#os.system("convert_dir_to_note_sequences --input_dir=\"" + INPUT_DIRECTORY + "\" --output_file=\"" + SEQUENCES_TFRECORD + "\" --recursive")

#Trains a .mag
#os.system("polyphony_rnn_create_dataset --input=\"" + SEQUENCES_TFRECORD + "\" --output_dir=\"" + SEQUENCE_EXAMPLES + "\" --eval_ratio=0.10")


# 2) Training a model / generating a .mag file
# this code fails, with a very long error, and generates some graphs and stuff, but fails to generate a mag file and crashes the compiler
#os.system("polyphony_rnn_train --run_dir=\"/tmp/polyphony_rnn/logdir/run1\" --sequence_example_file=\"/tmp/polyphony_rnn/sequence_examples/training_poly_tracks.tfrecord\" --hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" --num_training_steps=20000")



# 3) Generating midi from model
#Runs from a .mag file
#BUNDLE_PATH = path + '/polyphony_rnn.mag'
#os.system("polyphony_rnn_generate --bundle_file=\"" + BUNDLE_PATH +  "\" --output_dir=\"" + FINAL_OUTPUT_DIR + "\" --primer_pitches=\"[67, 64, 60]\"")


SEQ_LEN = 32
NUM_MIDI_VALS = 128

# define batch size
BATCH_SIZE = 64

# define universal key
MASTER_MAJOR_KEY = 'C' # C major
MASTER_MINOR_KEY = 'A' # A minor

SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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

def prepreocessing():
    main_github_dir = os.getcwd() # get cwd
    input_dir = main_github_dir + '/Piano_MIDI_Files/Piano_E_Competition_2011/Piano_E_Competition_2011_1/' # get midi folder

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
                print("filename: ", filename)
                key, keyinfo = keyfindingalg.get_key(str(filepath))
                letter_list = key[0].split()
                letter = letter_list[0]
                if (keyinfo == 'major'):
                    master_index = SCALE.index(MASTER_MAJOR_KEY)
                else:
                    master_index = SCALE.index(MASTER_MINOR_KEY)
               
                current_index = SCALE.index(letter)
                transpose_int = master_index - current_index

                #print("curr key: ", key, "master: ", master_index)

                #dataframe_of_notes = extract_notes(note_sequence)
                #pitches = list(dataframe_of_notes['pitch'])

                # get min and max pitches
                #min_pitch = min(pitches)
                #max_pitch = max(pitches)
                print("curr key: ", letter, ", transposed key: ", SCALE[master_index])
                print(filename + ": transpose int: ", transpose_int)
                #print(filename, key, keyinfo)
                
                

                    

                # transpose to universal key
                if (transpose_int == 0): # means file is already in master major or minor key
                    #print("Gurrrrl, you got da master key")
                    continue

                # transpose 

                transposed = transpose(note_sequence, transpose_int)

                note_seq.sequence_proto_to_midi_file(transposed, 'transpositions/transposed-' + filename)

                print(type(transposed))

                #key, keyinfo = keyfindingalg.get_key(str(filepath))
                #letter = key[0][0]
                #print(letter)

                
                """ for key in notes:
                    if key == 'pitch':
                        #print(notes[key])
                print('----------------------') """

                # getting large collection of all the notes in all files
                #print('--------------------')
                #seq = extract_notes(transposed)
                #all_note_sequences.append(seq)


prepreocessing()

#note Sequences
os.system("convert_dir_to_note_sequences \
    --input_dir=\"./transpositions\" \
    --output_file=\"./tmp/notesequences.tfrecord\" \
    --recursive")

#Train Model
os.system("polyphony_rnn_create_dataset \
    --input=\"./tmp/notesequences.tfrecord\" \
    --output_dir=\"./tmp/polyphony_rnn/sequence_examples\" \
    --eval_ratio=0.10")

os.system("polyphony_rnn_train \
--run_dir=\"./tmp/polyphony_rnn/logdir/run1\" \
--sequence_example_file=\"./tmp/polyphony_rnn/sequence_examples/training_poly_tracks.tfrecord\" \
--hparams=\"batch_size=64,rnn_layer_sizes=[32,32]\" \
--num_training_steps=1000")


os.system("polyphony_rnn_generate \
    --run_dir=\"./tmp/polyphony_rnn/logdir/run1\" \
    --hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" \
    --output_dir=\"./tmp/polyphony_rnn/generated\" \
    --num_outputs=1 \
    --num_steps=128 \
    --primer_pitches=\"[67,64,60]\" \
    --condition_on_primer=true \
    --inject_primer_during_generation=false")

"""
#Generate Mag File
os.system("polyphony_rnn_generate \
--run_dir=\"./tmp/polyphony_rnn/logdir/run1\" \
--hparams=\"batch_size=64,rnn_layer_sizes=[32,32]\" \
--bundle_file=\"./tmp/rnn.mag\" \
--save_generator_bundle=True")

#Use MAg File
os.system("polyphony_rnn_generate \
--bundle_file=\"./tmp/rnn.mag\" \
--output_dir=\"./tmp/polyphony_rnn/generated\" \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches=\"[67,64,60]\" \
--condition_on_primer=True \
--inject_primer_during_generation=False")
"""