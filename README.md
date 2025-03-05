# SSR_VibraProfiler

SSR_VibraProfiler is a tool designed to establish a DNA fingerprint database based on whole genome SSR characteristics. It is used to predict varieties for unknown individuals. The software consists of four main parts: model construction, model evaluation, individual variety identification, and cross-validation.



## Installation
### Docker Installation
```bash
docker pull oldcat931/ssr_vibraprofiler:1.0
```
### Installation from source code or conda

#### Requirements

- **minia**  
- **misa** (only misa.pl is needed)
- **Python 3.9.1**
- **conda** (version 23.9.0, though other versions may work)

Make sure all dependencies are installed and added to the environment variables.

#### Conda Installation

```bash
conda install -c oldcat931 ssr_vibraprofiler
```

#### Installation from Source Code
```bash
git clone https://github.com/Olcat35412/SSR_VibraProfiler
cd SSR_VibraProfiler
conda env create -f SSR_VibraProfiler.yml
conda activate ssr_vibraprofiler_env
chmod +x *
# Manually add this directory to the environment variable
```

## Usage


### 1. SSR_VibraProfiler_model_build.py

This script integrates three key steps:

-Assembly of sequencing data using Minia

-Extraction of SSR information using MISA

-Screening of SSR information and model construction

Essential Parameters:
```
-s: Specifies the start stage.
-e: Specifies the end stage.
-pp: Indicates the polymorphism (as described in the main text of the paper).
-o: Output path for storing the final identification model and SSR lists.
-index: The index file. Contains two columns: the variety and the absolute path for the input file of Minia.


SSR_VibraProfiler_model_build.py -s 1 -e 3 -pp 0.8 -index index_file -o output_path
```

Optional Parameters:
```
-misap: Specifies the number of MISA processes (default: 1)
-miniap: Specifies the number of Minia processes (default: 1)
-miniac: Specifies the number of cores for Minia processes (default: 1)
```

### 2. SSR_VibraProfiler_evaluation.py

This script evaluates the model built by SSR_VibraProfiler_model_build.py.

Essential Parameters:

```
-index_file: Specifies the index file (same as used in model build).
-i: Specifies the input file generated by the model build step.


SSR_VibraProfiler_evaluation.py -index index_file -i polymorphism_ssr.csv
```
Optional Parameters:
```
-c: Specifies the color, this file contains one RGB color per line. Please ensure that the total number of colors corresponds to the total number of varieties.
```
### 3. SSR_VibraProfiler_model_predict.py
This script is used to predict the variety of an individual based on SSR information.

Essential Parameters:
```
-i: Specifies the SSR information file.
-d: Folder containing the KNN-based model file and filtered SSR list.
-index: Same index file used in previous steps.
-k: Number of nearest neighbors (K in KNN).

SSR_VibraProfiler_model_predict.py -i W-1-10.contigs.fa.misa -d /out/index_file_6_27.txt_0.8 -index index_file -k 37
```

### 4. SSR_VibraProfiler_cross_validation.py

This script performs cross-validation on the model built.

Essential Parameters:

```
-index: Refers to the same index file used in earlier steps.
-pp: Specifies the polymorphism (as described in the paper).
-o: Output directory path.
-log: File path to save result data.

SSR_VibraProfiler_cross_validation.py -index -pp -o -log
```


## Operation Example in Rhododendron
We used this software on 40 individuals from 8 Rhododendron varieties. You can follow these steps to replicate our work or use the software for your own research.



### Dataset

You can download the data from the following links:

Original Dataset: Approximately 800GB of sequencing data after decompression.
```
wget "https://china.scidb.cn/download?fileId=2779cd477eecc01f550144809f6b9e9a&username=jiangchen1234@163.com&traceId=a62cbd41-3e6d-482f-a101-6c2e9d1b8967"
```
Lightweight Dataset: Approximately 40GB after decompression (5% of the original dataset).
```
wget https://china.scidb.cn/download?fileId=b463c9e8f1e2ce27ccc770739e662d15&username=jiangchen1234@163.com&traceId=jiangchen1234@163.com
```

