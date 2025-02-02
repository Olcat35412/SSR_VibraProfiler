#!/usr/bin/env python3
import os
import argparse
parser = argparse.ArgumentParser(description='A Variety Recognition Model Based on Whole Genome SSR Digital Features')
parser.add_argument('-index','--index_path',required=True)
parser.add_argument('-pp',required=True)
parser.add_argument('-o','--output_path',required=True)
parser.add_argument('-log','--log_path',required=True)
args = vars(parser.parse_args())
index_file=os.path.abspath(args['index_path'])
pp=args['pp']
output_path=os.path.abspath(args['output_path'])
log_path=os.path.abspath(args['log_path'])

def generate_index_file_for_all_indivduals(index_file):
    new_name_list=[]
    varieties=[]
    with open (index_file) as f:
        lines=f.readlines()
    for i,line in enumerate(lines):
        line_content = line.strip()
        new_index_file_name = "{}.txt".format(line_content.split('\t')[1])
        new_index_varieties=line_content.split('\t')[0]
        new_name_list.append(new_index_file_name)
        varieties.append(new_index_varieties)
        with open(new_index_file_name, 'w') as new_file:
            for j, other_line in enumerate(lines):
                if i != j:
                    new_file.write(other_line)
    return varieties,new_name_list
varieties,all_index=generate_index_file_for_all_indivduals(index_file)
print(varieties,all_index)
paired_list = list(zip(varieties,all_index))
k=len(all_index)-1
for variety,index in paired_list:
    index_as_path=os.path.basename(index)
    storage_name=index
    directory_path=output_path+'/'+index_as_path+'_'+str(pp)
    command="SSR_VibraProfiler_model_build.py -s 3 -e 3 -pp {} -index {} -o {}".format(pp,index,output_path)
    print(command)
    os.system(command)
    command="SSR_VibraProfiler_model_predict.py -index {} -d {} -k {} -i {}>>{}".format(index,directory_path,k,index.split('.')[0]+".contigs.fa.misa",log_path)
    print(command)
    os.system(command)
