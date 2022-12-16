import rtmidi
import mido
import os
import time

from mido import Message,MidiFile,MidiTrack
from datetime import datetime, timedelta

#writes all messages as on and off messages and creates a Mido MIDI file
def write_file(events):
    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(events)
    mid.save("new_file.mid")

def main():
    instrument = mido.get_input_names()
    inport = mido.open_input(instrument[0])
    events = []

    #opens mido port and adds each message to events
    with mido.open_input(instrument[0]) as inport:
        #starts timer as each message ends
        start = datetime.now()    

        for message in inport:

            #how we get out of the read loop with no user input
            if message.note == '21':
                break
            
            #stops timer, counting the delta of ticks and assigning time
            end = datetime.now()
            ticks = ((end-start).microseconds)
            msg = message.copy(time=ticks)

            #adds to events which can be plugged into key detection
            events.append(msg)
        
            if int(ticks//1000000) % 10 == 0:
                write_file(events)
    
    print(events)

main()
