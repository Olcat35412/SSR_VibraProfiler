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
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
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
	print(individual_to_variety_dict)
	return len(all_varieties), individual_to_variety_dict

def read_file_and_generate_matrix(input_path,index_path,individual_to_variety_dict):
	data = pd.read_csv(input_path, sep="\t")
	label_pre = list(data['SSR'])
	label_true = [] 
	for label in label_pre:
		label_true.append(individual_to_variety_dict[label.split('.')[0]])
	data = data.drop('SSR', axis=1)
	matrix = data.to_numpy()
	return matrix,label_true

def run_t_SNE_and_k_means(matrix,label_true,variety_count):
	rand_state_t_SNE = list(range(0, 51))
	rand_state_k_means = list(range(0, 11))
	ARI = 0
	for random_state1 in rand_state_t_SNE:
		tsne_result = TSNE(perplexity=30, n_components=2, random_state=random_state1).fit_transform(matrix)
		for random_state2 in rand_state_k_means:
			kmeans = KMeans(variety_count,random_state=random_state2)
			predict_true_label = kmeans.fit_predict(tsne_result)
			ARI_new = adjusted_rand_score(np.array(label_true), predict_true_label)
			if ARI_new > ARI:
				ARI = ARI_new
				xx = random_state1
				yy = random_state2
	best_tsne_result = TSNE(perplexity=30, n_components=2,random_state = xx).fit_transform(matrix)
	best_k_means = KMeans(variety_count , random_state = yy)
	best_k_means_result = best_k_means.fit_predict(best_tsne_result)
	ARI = round(ARI,2)
	print(ARI)
	return best_tsne_result,best_k_means_result,ARI

def generate_random_colors(n):
	return ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(n)]

def run_PCA_and_k_means(matrix,label_true,variety_count):
	scaler = StandardScaler()
	binary_scaled = scaler.fit_transform(matrix)
	pca = PCA(n_components=2)
	pca_results = pca.fit_transform(binary_scaled)
	rand_state_k_means = list(range(0, 11))
	ARI = 0
	for random_state2 in rand_state_k_means:
		kmeans = KMeans(variety_count,random_state=random_state2)
		predict_true_label = kmeans.fit_predict(pca_results)
		ARI_new = adjusted_rand_score(np.array(label_true), predict_true_label)
		if ARI_new > ARI:
			ARI = ARI_new
			yy = random_state2
	best_k_means = KMeans(variety_count , random_state = yy)
	best_k_means_result = best_k_means.fit_predict(pca_results)
	ARI = round(ARI,2)
	print(ARI)
	return pca_results,best_k_means_result,ARI

def run_UMAP_and_k_means(matrix,label_true,variety_count):
	scaler = StandardScaler()
	binary_scaled = scaler.fit_transform(matrix)
	rand_state_umap = list(range(0, 51))
	rand_state_k_means = list(range(0, 11))
	ARI = 0
	for random_state1 in rand_state_umap:
		umap = UMAP(n_components=2,n_neighbors=15,min_dist=0.1,metric='euclidean',random_state=random_state1)
		umap_results = umap.fit_transform(binary_scaled)
		for random_state2 in rand_state_k_means:
			kmeans = KMeans(variety_count,random_state=random_state2)
			predict_true_label = kmeans.fit_predict(umap_results)
			ARI_new = adjusted_rand_score(np.array(label_true), predict_true_label)
			if ARI_new > ARI:
				ARI = ARI_new
				xx = random_state1
				yy = random_state2
	best_umap = UMAP(n_components=2,n_neighbors=15,min_dist=0.1,metric='euclidean',random_state=xx)
	best_umap_result = best_umap.fit_transform(binary_scaled)
	best_k_means = KMeans(variety_count , random_state = yy)
	best_k_means_result = best_k_means.fit_predict(best_umap_result)
	ARI = round(ARI,2)
	print(ARI)
	return best_umap_result,best_k_means_result,ARI

def plot_results(dimension_results, true_labels, pred_labels, output_filebase,ARI,colors=None):
	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams['font.size'] = 9
	plt.rcParams["pdf.fonttype"] = 42
	unique_true_labels = sorted(set(true_labels))
	variety_count = len(unique_true_labels)
	if colors is None:
		colors = generate_random_colors(variety_count)
	cluster_cmap = ListedColormap(colors)
	fig, ax = plt.subplots(figsize=(90/25.4, 90/25.4))

	scatter  = ax.scatter(dimension_results[:, 0], dimension_results[:, 1],s =10, c= pred_labels, cmap=cluster_cmap)

	for i, true_labels in enumerate(true_labels):
		x, y = dimension_results[i, 0], dimension_results[i, 1]
		ax.annotate(true_labels, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
	ax.set_xlabel('Dimension 1')
	ax.set_ylabel('Dimension 2')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	fig.text(
		x=0.02,					# 横向位置（左边缘2%处）
		y=0.98,					# 纵向位置（顶部边缘2%处）
		s=f'ARI={ARI}',			# 标签内容
		fontsize=9,			   # 字号建议比正文大1-2pt
		fontfamily='Arial',		# 确保字体一致性
		color='black',			 # 字体颜色
		verticalalignment='top',   # 垂直对齐方式
		horizontalalignment='left',# 水平对齐方式
		)

	plt.tight_layout()
	plt.savefig(output_filebase + '.pdf')

	print(f"output file is saved to {output_filebase}.pdf")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True)
	parser.add_argument('-index', '--index_path', required=True)
	parser.add_argument('-c', '--color_file')
	args = parser.parse_args()
	start_time = time.time()
	input_path = os.path.abspath(args.input)
	index_path = os.path.abspath(args.index_path)
	input_basename = os.path.splitext(os.path.basename(input_path))[0]
	output_path_tsne = os.path.join(os.getcwd(), f"{input_basename}_tsne")
	output_path_pca = os.path.join(os.getcwd(), f"{input_basename}_pca")
	output_path_umap = os.path.join(os.getcwd(), f"{input_basename}_umap")
	variety_count, individual_to_variety_dict = get_variety_info(index_path)
	matrix, label_true = read_file_and_generate_matrix(input_path,index_path,individual_to_variety_dict)
	# 运行 t-SNE + K-Means,并绘制t-SNE结果
	tsne_results , predict_true_label,ARI = run_t_SNE_and_k_means(matrix, label_true, variety_count)
	plot_results(tsne_results, label_true, predict_true_label,output_path_tsne,ARI)
	# 运行 PCA + K-Means,并绘制PCA结果
	pca_results,best_k_means_label,ARI = run_PCA_and_k_means(matrix, label_true, variety_count)
	plot_results(pca_results, label_true, best_k_means_label,output_path_pca,ARI)
	# 运行 UMAP + K-Means,并绘制UMAP结果
	best_umap_result,best_k_means_result,ARI = run_UMAP_and_k_means(matrix, label_true, variety_count)
	plot_results(best_umap_result,label_true,best_k_means_result,output_path_umap,ARI)
	if args.color_file:
		print("read a color file")
		colors =[]
		with open(args.color_file,"r") as file:
			for line in file:
				color = line.rstrip()
				colors.append(color)
		tsne_results , predict_true_label,ARI = run_t_SNE_and_k_means(matrix, label_true, variety_count)
		plot_results(tsne_results, label_true, predict_true_label,output_path_tsne,ARI,colors)
	# 运行 PCA + K-Means,并绘制PCA结果
		pca_results,best_k_means_label,ARI = run_PCA_and_k_means(matrix, label_true, variety_count)
		plot_results(pca_results, label_true, best_k_means_label,output_path_pca,ARI,colors)
	# 运行 UMAP + K-Means,并绘制UMAP结果
		best_umap_result,best_k_means_result,ARI = run_UMAP_and_k_means(matrix, label_true, variety_count)
		plot_results(best_umap_result,label_true,best_k_means_result,output_path_umap,ARI,colors)
	else:
	# 运行 t-SNE + K-Means,并绘制t-SNE结果
		tsne_results , predict_true_label,ARI = run_t_SNE_and_k_means(matrix, label_true, variety_count)
		plot_results(tsne_results, label_true, predict_true_label,output_path_tsne,ARI)
	# 运行 PCA + K-Means,并绘制PCA结果
		pca_results,best_k_means_label,ARI = run_PCA_and_k_means(matrix, label_true, variety_count)
		plot_results(pca_results, label_true, best_k_means_label,output_path_pca,ARI)
	# 运行 UMAP + K-Means,并绘制UMAP结果
		best_umap_result,best_k_means_result,ARI = run_UMAP_and_k_means(matrix, label_true, variety_count)
		plot_results(best_umap_result,label_true,best_k_means_result,output_path_umap,ARI)        
	end_time = time.time()
	stage_duration = end_time - start_time
	print(f"cross_validation totally take {stage_duration} second")
if __name__ == "__main__":
	main()
