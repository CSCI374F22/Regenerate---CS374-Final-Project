# testkeyfindingloop.py

# test looping through key finding alg repeatedly through all the midi files

# referenced https://stackoverflow.com/questions/19587118/iterating-through-directories-with-python and https://www.geeksforgeeks.org/os-walk-python/
# to figure out recursively going through folders within folders

import math
import mido
import sys
import os
import keyfindingalg

def main():
    # loop through all the midi in the folder
    #path = os.getcwd() # get cwd
    os.chdir('..') # go up one directory
    main_github_dir = os.getcwd() # get cwd
    pianocompetitiondir = main_github_dir + '/PianoCompetitionMidi/'

    #Python method os.walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.

    for (root,dirs,files) in os.walk(pianocompetitiondir, topdown=True):
        # need all root, dirs, files in order to get the files alone
        #print ("root: ", root)
        #print ("dirs: ", dirs)
        #print ("files: ", files) # files are the actual midi files
        #print(files[0])
        # loop though all the files
        for filename in files:
            #print(filename)
            if len(dirs) == 0:
                filepath = root + '/' + filename
                #print(filepath)
                key = keyfindingalg.get_key(str(filepath))
                print(filename, key)
        print ('--------------------------------')

main()