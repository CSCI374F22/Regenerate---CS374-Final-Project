#attempting to generate a bundle file from a checkpoint in magenta

import magenta
import note_seq
#import os
#import numpy as np
#import pandas as pd
#import magenta.scripts.convert_dir_to_note_sequences as scripts
import tensorflow.compat.v2 as tf
from tensorboard.plugins.hparams import api as hp
#import random
#import keyfindingalg
#import math
#import mido

#from note_seq import sequences_lib
#from note_seq import chords_encoder_decoder
#from note_seq import chord_inference
#from tensorflow.data import Dataset, TFRecordDataset
#from note_seq import midi_io
#from note_seq import sequences_lib # https://github.com/magenta/note-seq/blob/a7ea6b3ce073b0791fc5e89260eae8e621a3ba0c/note_seq/chord_inference.py for quantization of note_seq
#from note_seq import chords_encoder_decoder
#from magenta.pipelines import note_sequence_pipelines

#import os


#from magenta.models.polyphony_rnn import polyphony_model
#from magenta.models.polyphony_rnn import polyphony_sequence_generator
from magenta.models.shared import sequence_generator
from magenta.models.shared import events_rnn_model
#from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
#from note_seq.protobuf import music_pb2
from magenta.contrib import training as contrib_training
from magenta.models.polyphony_rnn import polyphony_encoder_decoder




def generate_bundle(checkpoint_path):

    default_config = events_rnn_model.EventSequenceRnnConfig(
        details = generator_pb2.GeneratorDetails(
            id='Pls', 
            description='RNN by three tired undergrads'),
        encoder_decoder=note_seq.OneHotEventSequenceEncoderDecoder(
            polyphony_encoder_decoder.PolyphonyOneHotEncoding()),
        hparams=contrib_training.HParams(
            batch_size=64,
            rnn_layer_sizes=[2, 32, 3],
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.001))

    #Make Base Model
    mag_model = events_rnn_model.EventSequenceRnnModel(config=default_config)

    #MAke generator details
    deets = generator_pb2.GeneratorDetails(id='Pls', description='RNN for music made by oberlin undergrad')

    #Make Sequence Generator
    generator = sequence_generator.BaseSequenceGenerator(
        model = mag_model,
        details = deets,
        checkpoint = checkpoint_path,
        bundle = None
    )

    #Run create_generator_bundle() on object
    generator.create_bundle_file(
        bundle_file = './bundles',
        bundle_description = "JustPleaseWork"
    )



def main():
    generate_bundle('./training_checkpoints/')

main()