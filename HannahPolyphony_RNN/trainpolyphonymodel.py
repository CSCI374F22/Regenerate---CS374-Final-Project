# pipelinetest.py
# code referenced from https://github.com/magenta/magenta/tree/main/magenta/models/polyphony_rnn
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

from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train

from magenta.models.polyphony_rnn import polyphony_model
""" from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle """
""" import note_seq
from note_seq.protobuf import generator_pb2 """
from note_seq.protobuf import music_pb2


tf.disable_v2_behavior() # ???

#flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

num_training_steps = 2000 # not sure if that is enough or too much
batch_size = 128 # default batch size, Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models

# define the variable eval_ratio (which is the percentage of data used for evaluation/testing
# so if eval_ratio is 0.1, 10% of data will be used for testing, 90% for training)
tf.app.flags.DEFINE_float(
    'eval_ratio', 0.1,
    'Fraction of input to set aside for eval set. Partition is randomly '
    'selected.')

class MyPipeline(pipeline.Pipeline):

    def __init__(self):
        # will have to make all of our midi data into mido, possibly? Or not since processing midi
        super(MidiFile, nsPipeline).__init__(
            # set input_type to be midi since input is midi
            input_type = MidiFile,
            # set ouput_type to be Note Sequence since magenta uses note sequences
            # and we can easily convert between midi and note sequences
            ouput_type = music_pb2.NoteSequence
        )

# whether or not we are training, 
# test = False --> run with training data
# test = True --> run with testing data and don't update weights
# If True, this process only evaluates the model and does not update weights.'
#test = False

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # path to directory of midi files
    #midi_file_list = []
    path = '/Users/hannah/Documents/cs374/Final Proj/Regenerate---CS374-Final-Project'
    #dir_list = os.listdir(path)
    #counter = 0
    #open(dir_list[0])
    #print(dir_list)
    #for file_name in dir_list:
        # save midi file


        #s = file_name.strip('\"')
        #cleaned_name = ''.join(i for i in s if i not in '"')
    """print(counter)
        #open(file_name)
        midi_file = MidiFile(file_name)
        midi_file_list.append(midi_file)
        #print(counter)
        counter += 1 """
    #print("Files and directories in ", path, ":")
    #outputs = []

    # Change the current working directory to the directory containing all the midi files
    #os.chdir('8-bitGameSounds')

   # print(MyPipeline.output_type)
    """ for file_name in dir_list:
        midi_file = MidiFile(file_name)
        output = MyPipeline.transform(midi_file)
        print(output) """
            

    input_dir = path + '/Piano E-Competition 2011 (3)'
    tmp = path + '/tmp'

    tf.gfile.MakeDirs(tmp)

    output_dir = path + '/tmp/notesequences.tfrecord'

    #print(midi_file_list)
    #input_dir = path + '/8-bitGameSounds'
    #output_dir = path + '/tmp/notesequences.tfrecord'


    # Converts files to NoteSequences and writes output to notesequences.tfrecord
    # for all models creating the dataset is the same
    scripts.convert_directory(input_dir, output_dir, False) # false means no recursion, recursive = False, want recursive = True, when there are folders inside the directory containing relevant midi

    # create SequenceExamples, which are fed into the model during training and evaluation aka testing
    # Sequence Examples contain sequence of inputs and a sequence of labels that represent a polyphonic sequence


    # This next part is different depending on which model you use
    # In this example I'm using polyphony_rnn


    # The following code extracts polyphonic sequences from the tfrecord of NoteSequences and save them as SequenceExamples
    # Result: two collections of Sequence Examples, one for training, one for testing

    # define a pipeline_instance
    polyphony_pipeline_instance = polyphony_rnn_pipeline.get_pipeline(
        min_steps=80,  # 5 measures, might need to make longer to account for longer pieces (80 / 16)
        max_steps=512, # max 32 measures (512 / 16)
        eval_ratio=FLAGS.eval_ratio,
        config=polyphony_model.default_configs['polyphony']) # returns pipeline instance which creates the RNN dataset (partitions data into training and test/eval)

    polyphony_output_dir = path + '/tmp/polyphony_rnn/sequence_examples'
    
    # input will now be the tfrecord file containin ghe notesequences
    polyphony_input_dir = output_dir

    print(polyphony_input_dir)

    # iterate through the tfrecord which contains all the notesequences, and save that ouput to polyphony_output_dir
    # pipeline = polyphony_pipeline_instance
    # input_iterator = tf_record_iterator
        # - iterates over input data, items returned bu it are fed
        #   directly into pipeline's transform operator
    
    # creates wo collections of SequenceExamples will be generated, 
    # one for training, and one for evaluation, 
    # where the fraction of SequenceExamples in the evaluation set is determined by --eval_ratio

    """Runs the a pipeline on a data source and writes to a directory.
    Run the pipeline on each input from the iterator one at a time.
    A file will be written to `output_dir` for each dataset name specified
    by the pipeline. pipeline.transform is called on each input and the
    results are aggregated into their correct datasets."""

    pipeline.run_pipeline_serial(
        polyphony_pipeline_instance,
        pipeline.tf_record_iterator(polyphony_input_dir, polyphony_pipeline_instance.input_type),
        polyphony_output_dir)
    

    # don't get many verbose outputs like the command line stuff

    #successful up to this point
    
    #eval_dir = polyphony_output_dir + 'eval_poly_tracks.tfrecord'

    # the TFRecord file of SequenceExamples that will be fed to the model (training data)
    sequence_example_dir = polyphony_output_dir + 'training_poly_tracks.tfrecord'

    run_dir = path + '/tmp/polyphony_rnn/logdir/run1' # directory where checkpoints and TensorBoard data for this run will be stored

    # add two folders two run1 training (train) and testing (eval)
    training_dir = run_dir + '/train'
    eval_dir = run_dir + '/eval'

    # Returns a function that builds the TensorFlow graph.
    # mode: 'train', 'eval', or 'generate'. Only mode related ops are added to the graph
    build_graph_fn = events_rnn_graph.get_build_graph_fn(
      'train', polyphony_model.default_configs['polyphony'], sequence_example_dir)

    # make directory
    tf.gfile.MakeDirs(training_dir)
    # add logs
    tf.logging.info('Train dir: %s', training_dir)

    # train model
    #training(False, build_graph_fn, training_dir)

    # set test to true, so now goes through testing / eval data and evaluates model
    #test = True
    # now eval is true so will run testing / evaluation of model
    #testing(True, build_graph_fn, training_dir, eval_dir, sequence_example_dir)

def training(test, build_graph_fn, train_dir):
    if not test:  # if eval is false then will run training (eval always starts off as false)
        events_rnn_train.run_training(
            build_graph_fn, train_dir, num_training_steps
        ) # run training loop

def testing(test, build_graph_fn, train_dir, eval_dir, seq_file_path):
    # count_records counts numbr of items in the sequence_example
    # make directory
    if test:
        tf.gfile.MakeDirs(eval_dir)
        tf.logging.info('Eval dir: %s', eval_dir) # not necessary just wanna check
        num_batches = (magenta.common.count_records(seq_file_path)) // polyphony_model.default_configs['polyphony'].hparams.batch_size
        events_rnn_train.run_eval(build_graph_fn, train_dir, eval_dir, num_batches)






main()