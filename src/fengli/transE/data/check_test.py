from time import clock
import bisect 
import os

#check how many entities and relations of the testing file is in training file

def read_file(file_name):
    File = open(file_name,"r")
    lines = File.readlines()
    File.close()
    return lines

def main():

	cnt_entity = 0
	cnt_relation = 0

	lines_train_entity = read_file("entity2id.txt")
	lines_train_relation = read_file("relation2id.txt")
	lines_test = read_file("../test.txt")

	entity_train=[]
	relation_train=[]
	entity_test=[]
	relation_test=[]
	for line in lines_test:
		tmp = line.replace('\n','').split('\t')
		tmp_entity = []
		entity_test.append(tmp[0])
		for item in tmp[1].split(' '):
			tmp_entity.append(item)
		for item in tmp_entity:
			entity_test.append(item)
		relation_test.append(tmp[2])

	for line in lines_train_entity[1:]:
		tmp = line.replace('\n','').split('\t')      
		entity_train.append(tmp[0])

	for line in lines_train_relation[1:]:
		tmp = line.replace('\n','').split('\t')  
		relation_train.append(tmp[0])

	size_test_entity = len(entity_test)
	size_test_relation = len(relation_test)
	
	#print(len(entity_train))
	print(entity_train[1963128])
	for entity in entity_test:
		index = bisect.bisect_left(entity_train,entity)
		#print(index)
		#print(entity)
		#print(entity_train[index])
		if(entity != entity_train[index]):
			cnt_entity +=1
			print(entity)
	for relation in relation_test:
		index = bisect.bisect_left(relation_train,relation)
		if(relation != relation_train[index]):
			cnt_relation +=1
			print(relation)
	
	print("size_entity",size_test_entity)
	print("size_relation",size_test_relation)
	print("wrong entity",cnt_entity)
	print("wrong relation",cnt_relation)
	print("entity missing rate: ", str(float(cnt_entity)/size_test_entity*100), "%")
	print("relation missing rate: ", str(float(cnt_relation)/size_test_relation*100), "%")

main()
