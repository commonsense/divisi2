import json
from csc import divisi2

def main():
    stuff = []
    with open('usertest_data.txt') as file:
        for line in file:
            if line.strip():
               concept1, rel, concept2, val = line.strip().split('\t')
               val = int(val)
               stuff.append((val, concept1, ('right', rel, concept2)))
               stuff.append((val, concept2, ('left', rel, concept1)))
    matrix = divisi2.make_sparse(stuff)
    divisi2.save(matrix, 'usertest_data.pickle')

if __name__ == '__main__': main()

