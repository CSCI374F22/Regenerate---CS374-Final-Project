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

def generate_bundle():
    gen_details = generator_pb2.GeneratorDetails(
                id='polyphony_rnn',
                description='regenerate.')
    base_model = model.BaseModel()
    base_gen = sequence_generator.BaseSequenceGenerator(
        model= base_model,
        details = gen_details,
        checkpoint='training_checkpoints',
        bundle = None)
    base_gen.create_bundle_file('bundle_path',None)


def main():
    generate_bundle()

main()