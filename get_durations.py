#Program gets the sum of durations of all midi note_off messages, getting the total amount of time a note was played

import sys
from mido import Message,MidiFile,MidiTrack

#returns a list containing each possible value for MIDI note values of specific note for the first 8 octaves
#might later hard code these values in and delete function
def plus12(base_num):

    #init the return list with the number passed in
    base_adds = [base_num]

    #starting at 1, add 12 to each each step and append it to the list
    for i in range(1,8):
        next_num = base_num + (12*i)
        base_adds.append(next_num)

    return base_adds

def string_sub(str1,str2):
    if str2 in str1:
        str1.replace(str2,'')

def check_offs(filename):
    test = MidiFile(filename)
    mid = MidiFile(type=0)
    events = []
    i = 0
    for msg in test:
        if msg.type == 'note_off':
            i += 1

    if i == 0:
        for msg in test:
            events.append(msg)
            if msg.type == 'note_on':
                off = msg.copy()
                off.type = 'note_off'
                events.append(off)
    
        trackname = string_sub(filename,'Piano_MIDI_Files/Jazz_Piano/')
        track = MidiTrack(events)
        mid.tracks.append(track)
        mid.save(trackname)
    
    return [i,mid]



def scan(filename):
    returns = check_offs(filename)
    if returns[0] == 0:
        test = returns[1]
    else:
        test = MidiFile(filename)
     
    #note values start at 21(A0) and go to 127(G9), incrementing by 12 each octave
    note_dict = {'A': 0, 'A#': 0, 'B': 0,'C': 0, 'C#': 0, 'D': 0, 'D#': 0, 'E': 0, 'F': 0, 'F#': 0, 'G': 0, 'G#': 0}

    A_values = plus12(21)
    A_sharp_values = plus12(22)
    B_values = plus12(23)
    C_values = plus12(24)
    C_charp_values = plus12(25)
    D_values = plus12(26)
    D_sharp_values = plus12(27)
    E_values = plus12(28)
    F_values = plus12(29)
    F_sharp_values = plus12(30)
    G_values = plus12(31)

    for msg in test:
        print(msg)
        if msg.type == 'note_off':
            if msg.note in A_values:
                note_dict['A'] += msg.time
            elif msg.note in A_sharp_values:
                note_dict['A#'] += msg.time
            elif msg.note in B_values:
                note_dict['B'] += msg.time
            elif msg.note in C_values:
                note_dict['C'] += msg.time
            elif msg.note in C_charp_values:
                note_dict['C#'] += msg.time
            elif msg.note in D_values:
                note_dict['D'] += msg.time
            elif msg.note in D_sharp_values:
                note_dict['D#'] += msg.time
            elif msg.note in E_values:
                note_dict['E'] += msg.time
            elif msg.note in F_values:
                note_dict['F'] += msg.time
            elif msg.note in F_sharp_values:
                note_dict['F#'] += msg.time
            elif msg.note in G_values:
                note_dict['G'] += msg.time
            else:
                note_dict['G#'] += msg.time
    
    #in lieu of returning 
    """ for note in note_dict.keys():
        print(note,":",note_dict[note])
    print() """
    
    return note_dict

#pass in the name of the midi file without the .mid extension
""" def main():
    filename = sys.argv[1]
    note_dict = scan(filename)
    print(note_dict) """

#main()