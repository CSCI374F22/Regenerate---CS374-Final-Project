# had to create a separate file for generation since importing magenta.scripts.convert_dir_to_note_sequences
# caused a duplicate flag error with the code used for generation (imported as poly_gen)

import magenta
from magenta.pipelines import pipeline
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline as nsPipeline
from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.polyphony_rnn import polyphony_rnn_pipeline
from note_seq.protobuf import music_pb2
import tensorflow
from mido import MidiFile

# import magenta.scripts.convert_dir_to_note_sequences as scripts # can't have this or else duplicate flag error
import os

from note_seq import abc_parser
from note_seq import midi_io
from note_seq import musicxml_reader
import tensorflow.compat.v1 as tf

from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train

from magenta.models.polyphony_rnn import polyphony_model
import magenta.models.polyphony_rnn.polyphony_rnn_generate as poly_gen
from magenta.models.polyphony_rnn import polyphony_sequence_generator
""" from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle """
""" import note_seq
from note_seq.protobuf import generator_pb2 """
from note_seq.protobuf import music_pb2

def main():

        # get_checkpoint() - Get the training dir or checkpoint path to be used by the model

    #   """Polyphony RNN generation code as a SequenceGenerator interface."""
    path = '/Users/hannah/Documents/cs374/Final Proj/Testing'
    run_dir = path + '/tmp/polyphony_rnn/logdir/run1'
    training_dir = run_dir + '/train'

    generator = polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
        model=polyphony_model.PolyphonyRnnModel(polyphony_model.default_configs['polyphony']),
        details=polyphony_model.default_configs['polyphony'].details,
        steps_per_quarter=polyphony_model.default_configs['polyphony'].steps_per_quarter,
        checkpoint=training_dir,
        bundle=poly_gen.get_bundle())
    
    poly_gen.run_with_flags(generator)

main()