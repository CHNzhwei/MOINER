# IE-MOIF: a multi-omics integration framework with information enhancement and image representation learning
## IE-MOIF Introduction
![image](https://github.com/CHNzhwei/IE-MOIF/blob/master/IE-MOIF.png)
## Install
conda create -n IE-MOIF python=3.7<br>
conda activate IE-MOIF<br>
pip install -r requirements.txt<br>
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 –f https://download.pytorch.org/whl/torch_stable.html --user<br>
pip install ./utils_map/ lapjv-1.3.1.tar.gz<br>
## Usage
<b>Integration Step</b><br>
python main_IE-MOIF.py --data <i>omics1_file omics_file omics3_file</i> --label <i>label_file</i> --type <i>omics_1_name omics_2_name omics_3_name</i> --fs_num 1000 1000 500<br>
<b>Classification Step</b><br>
python VIT.py --task num_class --patch num_patch --mark dataset
## Example
1. python main_IE-MOIF.py --data ./example/mRNA.csv ./example/meth.csv ./example/miRNA.csv --label ./example/label.csv --type mRNA meth miRNA --drm fs --fs_num 1000 1000 500 --fem tsne<br>
2. python main_En-VIT.py --n_class 2 --patch 25




