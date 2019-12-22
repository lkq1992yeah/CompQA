from time import clock
import bisect 
import os

#split mulity e2 mulity tirples
 
def read_file(file_name):
    File = open(file_name,"r")
    lines = File.readlines()
    File.close()
    return lines

# def append_entitiy(line_list,entity_list):
#     if(line_list[0] not in entity_list):
#         entity_list.append(line_list[0])
#     if(line_list[2] not in entity_list):
#         entity_list.append(line_list[2])
#     return entity_list

# def append_relation(line_list,relation_list):
#     if(line_list[1] not in relation_list):
#         relation_list.append(line_list[1])
#     return relation_list



def write_file(entity_list,relation_list,triple_list,train_list,origin_file_name):
    
    if os.path.isdir("./new_transfered_"+origin_file_name):
        pass
    else:
        os.mkdir("./new_transfered_"+origin_file_name)

    file_name = "new_transfered_"+origin_file_name+"/entity2id.txt"
    File = open(file_name,"w")
    File.write(str(len(entity_list)))
    File.write('\n')
    for i in range(len(entity_list)):
        File.write(str(entity_list[i]))
        File.write('\t')
        File.write(str(i))
        File.write('\n')
    File.close()
    
    file_name = "new_transfered_"+origin_file_name+"/entity2id_forTest.txt"
    File = open(file_name,"w")
    for i in range(len(entity_list)):
        File.write(str(entity_list[i]))
        File.write('\t')
        File.write(str(i))
        File.write('\n')
    File.close()
    
    file_name = "new_transfered_"+origin_file_name+"/relation2id.txt"
    File = open(file_name,"w")
    File.write(str(len(relation_list)))
    File.write('\n')
    for i in range(len(relation_list)):
        File.write(str(relation_list[i]))
        File.write('\t')
        File.write(str(i))
        File.write('\n')
    File.close()

    file_name = "new_transfered_"+origin_file_name+"/relation2id_forTest.txt"
    File = open(file_name,"w")
    for i in range(len(relation_list)):
        File.write(str(relation_list[i]))
        File.write('\t')
        File.write(str(i))
        File.write('\n')
    File.close()
    
    file_name = "new_transfered_"+origin_file_name+"/triple2id.txt"
    File = open(file_name,"w")
    File.write(str(len(triple_list)))
    File.write('\n')
    for i in range(len(triple_list)):
        File.write(str(triple_list[i][0])+'\t'+str(triple_list[i][1])+'\t'+str(triple_list[i][2]))
        File.write('\n')
    File.close()

    file_name = "new_transfered_"+origin_file_name+"/train.txt"
    File = open(file_name,"w")
    #File.write(str(len(train_list)))
    #File.write('\n')
    for i in range(len(train_list)):
        File.write(str(train_list[i][0])+'\t'+str(train_list[i][1])+'\t'+str(train_list[i][2]))
        File.write('\n')
    File.close()



def main():

    file_name ="freebase-FB2M"  #original data name
    file_path = "origin_data/"+file_name+".txt"
    start=clock()

    File = open("log.txt","a")
    File.write("reading data\n")
    File.close()
    lines = read_file(file_path)

    entity=[]
    relation=[]
    triple=[]
    train=[]
    
    File = open("log.txt","a")
    File.write("processing data\n")
    File.close()
    for line in lines:
        tmp = line.replace("www.freebase.com",'').replace('\n','').split('\t')
        # entity = append_entitiy(tmp,entity)
        # relation = append_relation(tmp,relation)
        tmp_entity = []
        
        
        entity.append(tmp[0])

        for item in tmp[2].split(' '):
        	tmp_entity.append(item)
        for item in tmp_entity:
        	entity.append(item)

        # entity.append(tmp[2])
        relation.append(tmp[1])

    print("raw data divided into lists")

    for i in range(10):
        print(i," is ",entity[i])

    entity = {}.fromkeys(entity).keys()
    relation = {}.fromkeys(relation).keys()

    for i in range(10):
        print(i," is ",entity[i])


    entity.sort()
    relation.sort()

    for i in range(10):
        print(i," is ",entity[i])
    print("lists sorted")

    for line in lines:
        tmp = line.replace("www.freebase.com",'').replace('\n','').split('\t')
        tmp_entity = []
        
        for item in tmp[2].split(' '):
        	tmp_entity.append(item)

        e1 = tmp[0]
        rel = tmp[1]
        for item in tmp_entity:
        	e2 =  item
        	train.append([e1,e2,rel])

    for line in lines:
        tmp = line.replace("www.freebase.com",'').replace('\n','').split('\t')

        tmp_entity = []
        for item in tmp[2].split(' '):
        	tmp_entity.append(item)

        e1 = bisect.bisect_left(entity,tmp[0])
        # e2 = bisect.bisect_left(entity,tmp[2])
        rel = bisect.bisect_left(relation,tmp[1])
        for item in tmp_entity:
        	e2 = bisect.bisect_left(entity,item)
        	if(isinstance(e1, int) and isinstance(e2, int) and isinstance(rel, int)):
        		triple.append([e1,e2,rel])
        	else:
        		print("e1:",e1," e2:",e2," rel:",rel," are not int")
    
    File = open("log.txt","a")
    File.write("writing data\n")
    File.close()
    write_file(entity,relation,triple,train,file_name)

    finish=clock()
    File = open("log.txt","a")
    File.write(file_path+" processing time: "+ str(finish-start) + " seconds\n")
    File.close()

main()
