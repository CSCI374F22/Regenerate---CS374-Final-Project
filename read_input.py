import rtmidi
import mido
import os
import time

from mido import Message,MidiFile,MidiTrack

#writes all messages as on and off messages and creates a Mido MIDI file
running = True
#writes all messages as on and off messages and creates a Mido MIDI file
def write_file(events):
    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(events)
    mid.save("new_file.mid")

def main():
    # https://stackoverflow.com/questions/61814607/how-do-you-get-midi-events-using-python-rtmidi
    instrument = mido.get_input_names()
    inport = mido.open_input(instrument[0])
    events = []
    len_events_ago = 0
    running = True
    while running:
        seconds = time.process_time() #start timer
        print("time(s):", seconds)
        msg = inport.receive()
        msg = msg.copy(time=100)
        events.append(msg)
        #print("type: ", type(msg))
        #print("events: ", events)

        if int(seconds) % 10 == 0:
            write_file(events)

        print(msg)    
    
    


main()
