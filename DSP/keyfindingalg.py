# keyfindingalg.py

import math

# Krumhansl-Schmuckler key-finding algorithm, explained here http://rnhart.net/articles/key-finding/

base_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# each number was figured out by Krumhansl and Jessler from their studies and each number is associated
# with each pitch class (C, C#, D, E, F, F#, G, G#, A, A#, B)   
major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]


# same but for minor
minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def main():

    

    # Alim is making function to find total durations of each pitch class in midi
    pitch_classes = {}


    # using this to check if works
    # { C: 432
    #   C#: 231
    #   D: 0
    #   D#: 405      
    #   E: 12       
    #   F: 316      
    #   F#: 4       
    #   G: 126       
    #   G#: 612       
    #   A : 0     
    #   A# : 191  
    #   B : 1  }

    pitch_classes['C'] = 432
    pitch_classes['C#'] = 231
    pitch_classes['D'] = 0
    pitch_classes['D#'] = 405
    pitch_classes['E'] = 12
    pitch_classes['F'] = 316
    pitch_classes['F#'] = 4
    pitch_classes['G'] = 126
    pitch_classes['G#'] = 612
    pitch_classes['A'] = 0
    pitch_classes['A#'] = 191
    pitch_classes['B'] = 1

    C = base_scale
    C_sharp = get_key_list('C#')
    D = get_key_list('D')
    D_sharp = get_key_list('D#')
    E = get_key_list('E')
    F = get_key_list('F')
    F_sharp = get_key_list('F#')
    G = get_key_list('G')
    G_sharp = get_key_list('G#')
    A = get_key_list('A')
    A_sharp = get_key_list('A#')
    B = get_key_list('B')


    """ print(C_maj)
    print(C_sharp_maj)
    print(D_maj)
    print(D_sharp_maj)
    print(E_maj) """

    # define major list of tuples
    C_maj_list = get_key_tuple_list(C, pitch_classes, True) # pass in True for major key
    C_sharp_maj_list = get_key_tuple_list(C_sharp, pitch_classes, True)
    D_maj_list = get_key_tuple_list(D, pitch_classes, True)
    D_sharp_maj_list = get_key_tuple_list(D_sharp, pitch_classes, True)
    E_maj_list = get_key_tuple_list(E, pitch_classes, True)
    F_maj_list = get_key_tuple_list(F, pitch_classes, True)
    F_sharp_maj_list = get_key_tuple_list(F_sharp, pitch_classes, True)
    G_maj_list = get_key_tuple_list(G, pitch_classes, True)
    G_sharp_maj_list = get_key_tuple_list(G_sharp, pitch_classes, True)
    A_maj_list = get_key_tuple_list(A, pitch_classes, True)
    A_sharp_maj_list = get_key_tuple_list(A_sharp, pitch_classes, True)
    B_maj_list = get_key_tuple_list(B, pitch_classes, True)


    #define minor list of tuples
    C_min_list = get_key_tuple_list(C, pitch_classes, False) # pass in False for minor key
    C_sharp_min_list = get_key_tuple_list(C_sharp, pitch_classes, False)
    D_min_list = get_key_tuple_list(D, pitch_classes, False)
    D_sharp_min_list = get_key_tuple_list(D_sharp, pitch_classes, False)
    E_min_list = get_key_tuple_list(E, pitch_classes, False)
    F_min_list = get_key_tuple_list(F, pitch_classes, False)
    F_sharp_min_list = get_key_tuple_list(F_sharp, pitch_classes, False)
    G_min_list = get_key_tuple_list(G, pitch_classes, False)
    G_sharp_min_list = get_key_tuple_list(G_sharp, pitch_classes, False)
    A_min_list = get_key_tuple_list(A, pitch_classes, False)
    A_sharp_min_list = get_key_tuple_list(A_sharp, pitch_classes, False)
    B_min_list = get_key_tuple_list(B, pitch_classes, False)


    # make a giant list of all 24 major and minor correlation coeficents
    all_keys = []

    all_keys.append(('C maj', get_correlation_coefficient(C_maj_list)))
    all_keys.append(('C# maj' ,get_correlation_coefficient(C_sharp_maj_list)))
    all_keys.append(('D maj', get_correlation_coefficient(D_maj_list)))
    all_keys.append(('D# maj', get_correlation_coefficient(D_sharp_maj_list)))
    all_keys.append(('E maj', get_correlation_coefficient(E_maj_list)))
    all_keys.append(('F maj', get_correlation_coefficient(F_maj_list)))
    all_keys.append(('F# maj', get_correlation_coefficient(F_sharp_maj_list)))
    all_keys.append(('G maj', get_correlation_coefficient(G_maj_list)))
    all_keys.append(('G# maj', get_correlation_coefficient(G_sharp_maj_list)))
    all_keys.append(('A maj', get_correlation_coefficient(A_maj_list)))
    all_keys.append(('A# maj', get_correlation_coefficient(A_sharp_maj_list)))
    all_keys.append(('B maj', get_correlation_coefficient(B_maj_list)))

    all_keys.append(('C min', get_correlation_coefficient(C_min_list)))
    all_keys.append(('C# min', get_correlation_coefficient(C_sharp_min_list)))
    all_keys.append(('D min', get_correlation_coefficient(D_min_list)))
    all_keys.append(('D# min', get_correlation_coefficient(D_sharp_min_list)))
    all_keys.append(('E min', get_correlation_coefficient(E_min_list)))
    all_keys.append(('F min', get_correlation_coefficient(F_min_list)))
    all_keys.append(('F# min', get_correlation_coefficient(F_sharp_min_list)))
    all_keys.append(('G min', get_correlation_coefficient(G_min_list)))
    all_keys.append(('G# min', get_correlation_coefficient(G_sharp_min_list)))
    all_keys.append(('A min', get_correlation_coefficient(A_min_list)))
    all_keys.append(('A# min', get_correlation_coefficient(A_sharp_min_list)))
    all_keys.append(('B min', get_correlation_coefficient(B_min_list)))

    # Find key with highest correlation
    R = max(all_keys, key=lambda tup: tup[1])
    
    print(R)

# return a scale of a pitch class like the C# one: ['C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C']
def get_key_list(starting_note):
    start_idx = base_scale.index(starting_note)
    # begginning bit to add to end of scale
    wrap_around = base_scale[0: start_idx]

    partial_scale = base_scale[start_idx:]

    scale = partial_scale + wrap_around

    return scale

def get_key_tuple_list(scale, pitch_classes, major):

    # list of tuples
    T = []

    # iterate through scale (list)
    for i in range(len(scale)):
        note = scale[i]

        # get associated major profile
        if (major):
            profile_of_note = major_profile[i]
        else:
            profile_of_note = minor_profile[i]

        # get note duration value
        note_duration = pitch_classes[note]

        T.append((profile_of_note, note_duration))

    return T

def get_correlation_coefficient(coordinates):
    
    summation1 = 0

    avg_x = mean(coordinates, True) # gets avg of all x values 
    avg_y = mean(coordinates, False) # gets avg of all y values 

    for i in range(len(coordinates)):
        coordinate = coordinates[i]
        x = coordinate[0]
        y = coordinate[1]

        product = (x - avg_x) * (y - avg_y)

        summation1 += product
    
    summation2 = 0

    for i in range(len(coordinates)):
        coordinate = coordinates[i]
        x = coordinate[0]
        y = coordinate[1]
        product = math.pow((x - avg_x), 2)

        summation2 += product
    
    summation3 = 0

    for i in range(len(coordinates)):
        coordinate = coordinates[i]
        x = coordinate[0]
        y = coordinate[1]
        product = math.pow((y - avg_y), 2)

        summation3 += product
    
    R = summation1 / math.sqrt((summation2 * summation3))

    return R





# get mean of tuple list L, if first is true does mean of all of first elements, otherwise all of 2nd elements
def mean(L, first):
    summation = 0
    for item in L:
        if (first):
            summation += item[0]
        else:
            summation += item[1]
    
    return (summation / 12)
    




main()

