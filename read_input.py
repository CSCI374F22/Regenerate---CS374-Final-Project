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
            if message.note == 21:
                break
            
            #stops timer, counting the delta of ticks and assigning time
            end = datetime.now()
            #gets the difference from start to end in ms and converts to seconds
            miliseconds = (end-start).microseconds//1000
            seconds = (end-start).microseconds//1000000

            #converts seconds to ticks
            ticks = int(miliseconds // 2.604)
            msg = message.copy(time=ticks)
            print("ticks: ", ticks)
            print("msg: ", msg)

            #adds to events which can be plugged into key detection
            events.append(msg)

            #if its been 10 (20,30...) seconds then write the file to midi
            """ if seconds != 0 and seconds % 10 == 0:
                print("seconds: ", seconds)
                write_file(events) """
    
    write_file(events)       
    print(events)

main()
