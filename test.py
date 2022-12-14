import magenta
import sys
import os


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

#note Sequences
os.system("convert_dir_to_note_sequences \
    --input_dir=\"./Piano_MIDI_Files/\" \
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
--hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" \
--num_training_steps=20000")

#Generate Mag File
os.system("polyphony_rnn_generate \
--run_dir=\"./tmp/polyphony_rnn/logdir/run1\" \
--hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" \
--bundle_file=\"./tmp/rnn.mag\" \
--save_generator_bundle")

#Use MAg File
os.system("polyphony_rnn_generate \
--bundle_file=\"./tmp/rnn.mag\" \
--output_dir=\"./tmp/polyphony_rnn/generated\" \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches=\"[67,64,60]\" \
--condition_on_primer=true \
--inject_primer_during_generation=false")