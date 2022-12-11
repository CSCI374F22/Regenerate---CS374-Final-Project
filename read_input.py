import rtmidi
import mido
import os
import time

from mido import Message,MidiFile,MidiTrack

all_midi_events = []

#Pretty Self Explanatory
def set_port():
    midi_in = rtmidi.MidiIn()
    port = midi_in.get_ports() #should find 'USB MIDI Interface'
    midi_in.open_port(0)
    return port # port[0]

#Also pretty self explanatory
def close_port(midi_port):
    midi_port.close_port()

#Defines how it resonds to incoming midi events
#In this case its adds them to a list global to the function and prints them out for testing
def handle_input(event, data=None):
    message, deltatime = event
    all_midi_events.append(event)
    print('message:',message,"time:", deltatime, "data:", data)

#writes all messages as on and off messages and creates a Mido MIDI file
def write_file(midi_events):
    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(track)

    for event in midi_events:
        #set values for each on midi message
        on_msg = Message('note_on')
        note = event[0]

        on_msg.channel = note[0]
        on_msg.note = note[1]
        on_msg.velocity = note[2]
        on_msg.time = event[-1]

        #set values for each off midi message
        off_msg = Message('note_on')
        note = event[0]

        off_msg.channel = note[0]
        off_msg.note = note[1]
        off_msg.velocity = note[2]
        off_msg.time = event[-1]

        track.append(on_msg)
        track.append(off_msg)

    return mid
    

def main():
    midi_in = rtmidi.MidiIn() # used to be set_port()
    print(midi_in)  
    midi_in.set_callback(handle_input)  # this can't be called on a port only on rtmidi
    #waits 30 seconds; this will be called in a loop if there's still input in the master function
    time.sleep(30)
    
    #callback finishes
    midi_in.close_port()
    current = write_file(all_midi_events)


main()
