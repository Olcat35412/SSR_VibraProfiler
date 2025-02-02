#!/usr/bin/env python3
import random
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from operator import itemgetter
import os
from matplotlib.colors import ListedColormap
import argparse
parser = argparse.ArgumentParser(description='Validation of t-SNE and k-means')
parser.add_argument('-i','--input_path',required=True)
parser.add_argument('-index','--index_path',required=True)
args = vars(parser.parse_args())
input_path=os.path.abspath(args['input_path'])
index_path=os.path.abspath(args['index_path'])
plt.rcParams['font.size'] = 9
plt.rcParams["pdf.fonttype"] = 42
def get_variety_info(file_path):
    individual_variety_dict = dict()
    individual_to_variety_dict = dict()
    all_varieties = set()
    with open(file_path) as file:
        for line in file:
            line0 = line.rstrip().split('\t')
            variety = line0[0]
            individual = line0[1]
            all_varieties.add(variety)
            individual_to_variety_dict[individual] = variety
            if variety not in individual_variety_dict:
                individual_variety_dict[variety] = [individual]
            else:
                individual_variety_dict[variety].append(individual)
    return len(all_varieties), individual_to_variety_dict
variety_count, individual_to_variety_dict = get_variety_info(index_path)
plt.rcParams['font.size'] = 9
ssr_file_basename = os.path.basename(input_path)
data = pd.read_csv(input_path, sep="\t")
label_pre = list(data['SSR'])
label_true = []
for label in label_pre:
	label_true.append(individual_to_variety_dict[label.split('.')[0]])
headers = label_true
data = data.drop('SSR', axis=1)
matrix = data.to_numpy()
rand_state_t_SNE = list(range(0, 51))
rand_state_k_means = list(range(0, 11))
ARI = 0
ARI_State_LIST = []
for random_state1 in rand_state_t_SNE:
    tsne_result = TSNE(perplexity=30, n_components=2, random_state=random_state1).fit_transform(matrix)
    for random_state2 in rand_state_k_means:
        kmeans = KMeans(variety_count,random_state=random_state2)
        predict_ture_label = kmeans.fit_predict(tsne_result)
        ARI_new = adjusted_rand_score(np.array(label_true), predict_ture_label)
        ARI_State_LIST.append((str(random_state1) + ', '+str(random_state2), ARI_new))
        if ARI_new > ARI:
            ARI = ARI_new
            xx = random_state1
            yy = random_state2

print("The best ARI is:"+str(ARI))

ARI_State_LIST = sorted(ARI_State_LIST, key=itemgetter(1), reverse=True)
ARI_State_LIST_8 = []

ARI_State_LIST = enumerate(ARI_State_LIST)

for x, y in ARI_State_LIST:
    if x < 8:
        ARI_State_LIST_8.append(y)


state = []
ARI_list = []

for (a, b) in ARI_State_LIST_8:
    state.append(a)
    ARI_list.append(b)
    print(a)

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(184/25.4, 84/25.4))
ax1.bar(state, ARI_list, color='#2a9d8f')
ax1.yaxis.grid(True, linestyle='--', color='gray')
ax1.set_xticklabels(state, fontsize=7)
ax1.set_ylabel('Adjusted rand index (ARI)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i, v in enumerate(ARI_list):
    ax1.text(i, v, f"{v:.2}", ha='center', va='bottom')

# generate colors
def generate_random_colors(n):
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(n)]
    return colors
colors = generate_random_colors(variety_count)
cluster_cmap = ListedColormap(colors)

tsne = TSNE(n_components=2, random_state=xx, perplexity=30)
tsne_results = tsne.fit_transform(matrix)
kmeans = KMeans(variety_count, random_state=yy)
data['cluster'] = kmeans.fit_predict(tsne_results)

ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data['cluster'], cmap=cluster_cmap)

for i, header in enumerate(headers):
    x, y = tsne_results[i, 0], tsne_results[i, 1]
    ax2.annotate(header, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig(ssr_file_basename + '.pdf')
print("output pdf file saved in this file")
print(ssr_file_basename + '.pdf')
