import os
import numpy as np

def read_file(file_name):
    File = open(file_name,"r")
    lines = File.readlines()
    File.close()
    return lines

def main():

    lines = read_file("./entity2vec.vec")
    list_entity_vec_matrix = []
    for line in lines:
        tmp = line.replace('\n','').split('\t')
        tmp_vector = []
        
        for num in tmp[:-1]:
        	tmp_vector.append(float(num))
        list_entity_vec_matrix.append(tmp_vector)

    np_entity_vec_matrix = np.array(list_entity_vec_matrix)

    print(np_entity_vec_matrix.shape)

    lines = read_file("./relation2vec.vec")
    list_relation_vec_matrix = []
    for line in lines:
        tmp = line.replace('\n','').split('\t')
        tmp_vector = []
        
        for num in tmp[:-1]:
        	tmp_vector.append(float(num))
        list_relation_vec_matrix.append(tmp_vector)

    np_relation_vec_matrix = np.array(list_relation_vec_matrix)

    print(np_relation_vec_matrix.shape)

    return np_entity_vec_matrix,np_relation_vec_matrix
   