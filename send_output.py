import mido
import os
import time
import sys

from mido import Message,MidiFile,MidiTrack

def main():
    mid_file = sys.argv[1]
    instrument = mido.get_output_names()
    outport = mido.open_output(instrument[0])
    
    for msg in MidiFile(mid_file).play():
        outport.send(msg)

main()