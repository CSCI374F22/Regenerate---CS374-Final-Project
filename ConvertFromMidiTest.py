import os
import magenta
from magenta.pipelines import pipeline
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline as nsPipeline
from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.polyphony_rnn import polyphony_rnn_pipeline
from note_seq.protobuf import music_pb2
import tensorflow
from mido import MidiFile

import magenta.scripts.convert_dir_to_note_sequences as scripts
import os

from note_seq import abc_parser
from note_seq import midi_io
from note_seq import musicxml_reader
import tensorflow.compat.v1 as tf

def main():
    path = '/Users/hannah/Documents/cs374/Final Proj/Regenerate---CS374-Final-Project'
    input_dir = path + '/Piano E-Competition 2011 (3)'
    tmp = path + '/tmp'
    # make directory
    tf.gfile.MakeDirs(tmp)

    output_dir = path + '/tmp/notesequences.tfrecord'

    # Converts files to NoteSequences and writes output to notesequences.tfrecord
    # for all models creating the dataset is the same
    scripts.convert_directory(input_dir, output_dir, True) # false means no recursion, recursive = False, want recursive = True, when there are folders inside the directory containing relevant midi

    # only issue is there is no way to test if this worked

    # parse tfrecord file
    #raw_dataset = tf.data.TFRecordDataset(output_dir)
    #pipeline.tf_record_iterator(output_dir, music_pb2.NoteSequence)

    #tfrecord = tf.data.TFRecordDataset(output_dir)
    #iterator = iter(tfrecord)

    #item = iterator.next()
    


main()