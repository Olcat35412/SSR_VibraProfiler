#!/usr/bin/env python3
import pickle
import argparse
import pandas as pd
from sklearn import datasets
import os
import sys
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from functools import reduce
from operator import and_
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from operator import itemgetter
import matplotlib.pyplot as plt
import pickle
parser = argparse.ArgumentParser(description='A Variety Recognition Model Based on Whole Genome SSR Digital Features')
parser.add_argument('-d','--directory_path',required=True)
parser.add_argument('-i','--input_path',required=True)
parser.add_argument('-index','--index_path',required=True)
parser.add_argument('-k','--k_number',required=True)
args = vars(parser.parse_args())
directory_path=os.path.abspath(args['directory_path'])
input_path=os.path.abspath(args['input_path'])
list_save_path=directory_path+'/'+'polymorphism_ssr_list.pkl'
index_path=os.path.abspath(args['index_path'])
model_path=directory_path+'/predict_model.pkl'
k_number=args['k_number']
def get_individual_variety_dict(file_path):
    individual_variety_dict=dict()
    index_variety_dict=dict()
    all_individual=[]
    variety_list=[]
    with open(file_path) as file:
        index=0
        for line in file:
            line0=line.rstrip().split('\t')
            variety=line0[0]#get variety
            individual=line0[1]#get individual name
            index_variety_dict[index]=variety
            all_individual.append(individual)
            variety_list.append(variety)
            index+=1
            if variety not in individual_variety_dict:
                individual_variety_dict[variety]=[individual]
            else:
                individual_variety_dict[variety].append(individual)
    return individual_variety_dict,all_individual,variety_list,index_variety_dict
individual_variety_dict,all_individual,variety_list,index_variety_dict=get_individual_variety_dict(index_path)
#print('index_variety',index_variety_dict)
#print(individual_variety_dict,all_individual,variety_list)
with open(list_save_path, 'rb') as f:
    ssr_list = pickle.load(f)
with open(model_path,'rb') as f:
    model=pickle.load(f)
def get_ssr_from_file(misafile_path):
    ssr_dict = dict()
    ssr_dict2 = dict()
    ssr_list=[]
    with open(misafile_path) as file:
        for line in file:
            if line.count ('(') == 1:
                r_part = line.split('(',1)[1]
                motifs = r_part.split(')',1)[0]
                motifs_hh = r_part.split(')',1)[1]
                motifs_n = motifs_hh.split('\t',1)[0]
                intact_ssr = '('+ motifs + ')'+motifs_n
                ssr_list.append(intact_ssr)
    ssr_number=0
    for ssr in ssr_list:
        ssr_number=ssr_number+1
        if ssr not in ssr_dict:
            ssr_dict[ssr]=1
    for ssr in ssr_dict:
        ssr_dict2[ssr]=(ssr_dict[ssr])/(ssr_number)
    return ssr_dict
ssr_dict=get_ssr_from_file(input_path)
ssr_list2=[]
for ssr in ssr_list:
    if ssr not in ssr_dict:
        ssr_list2.append(0)
    else:
        ssr_list2.append(ssr_dict[ssr])
ssr_list2=np.array(ssr_list2)
#print(ssr_list2.reshape(1,-1))
c=model.predict(ssr_list2.reshape(1,-1))
c2=model.predict_proba(ssr_list2.reshape(1,-1))
a,b=model.kneighbors(ssr_list2.reshape(1,-1),n_neighbors=int(k_number))
#print(a,b)
variety2_list=[]
for listd in b:
    for item in listd:
        variety2_list.append(index_variety_dict[item])
count=0
for item in variety2_list:
    if item==c:
        count+=1
#print(index_variety_dict)
#reliability=(count/len(variety2_list))
#print(c)
#print(c2)
#print(variety2_list)
string1 = '\t'.join(variety2_list)
string2=str(os.path.basename(input_path))
result=string2+'\t'+string1
print(result)
#command='echo '+result+' >>/Rho/pycat/output.txt'
#os.system(command)
