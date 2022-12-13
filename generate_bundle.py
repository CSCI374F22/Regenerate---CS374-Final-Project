import magenta
import note_seq
import abc

from note_seq.protobuf import generator_pb2
from magenta.models import polyphony_rnn
from magenta.models.shared import model
from magenta.models.polyphony_rnn import polyphony_lib
from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.polyphony_rnn.polyphony_lib import PolyphonicEvent
from magenta.models.polyphony_rnn import polyphony_sequence_generator

from note_seq import chords_encoder_decoder
from magenta.pipelines import note_sequence_pipelines
from magenta.models.shared import sequence_generator

checkpoint_file = 'training_checkpoints'
gen_details = generator_pb2.GeneratorDetails(
                id='polyphony_rnn',
                description='regenerate.')

def generate_bundle():

    base_gen = sequence_generator.BaseSequenceGenerator(
        model= model.BaseModel,
        details = gen_details,
        checkpoint = checkpoint_file,
        bundle = None)

    print("base_gen is type:",type(base_gen))

    poly_gen = polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
        model = base_gen,
        details = gen_details,
        checkpoint = checkpoint_file, 
        bundle = None)

    print("poly_gen is type:",poly_gen)


    poly_gen.create_bundle_file('bundle_path',None)

def main():
    generate_bundle()

main()