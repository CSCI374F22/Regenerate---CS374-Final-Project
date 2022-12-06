import magenta
import sys
import os


# 1) Converting Midi to note sequences

# https://github.com/magenta/magenta/blob/main/magenta/scripts/README.md
INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>
# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

os.system("convert_dir_to_note_sequences --input_dir=\"" + INPUT_DIRECTORY + "\" --output_file=$SEQUENCES_TFRECORD --recursive")

#Trains a .mag
os.system("polyphony_rnn_create_dataset --input=\"/tmp/notesequences.tfrecord\" --output_dir=\"./tmp/polyphony_rnn/sequence_examples\" --eval_ratio=0.10")


# 2) Training a model / generating a .mag file
os.system("polyphony_rnn_train --run_dir=/tmp/polyphony_rnn/logdir/run1 --sequence_example_file=/tmp/polyphony_rnn/sequence_examples/training_poly_tracks.tfrecord --hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" --num_training_steps=20000")


# 3) Generating midi from model
#Runs from a .mag file
BUNDLE_PATH = "/Users/milesberry/Desktop/22-fall/Machine Learning/Generate---CS374-Final-Project/polyphony_rnn.mag"
os.system("polyphony_rnn_generate --bundle_file=\"" + BUNDLE_PATH +  "\" --output_dir=\"./tmp/polyphony_rnn\" --primer_pitches=\"[67, 64, 60]\"")
