# get_midi_messages.pu

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

# takes in a list of midi messages
def scan(msg_list):
    # returns = check_offs(filename)
    # if returns[0] == 0:
    #     test = returns[1]
    #     print("Was Empty")
    # else:
    #     print("Was not Empty")
    
    """ test = MidiFile(filename) """
     
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

    for msg in msg_list:
        # print(msg)
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