# IE-MOIF: an information enhancement and image-like representation-based framework for multi-omics integration
## IE-MOIF Introduction
![image](https://github.com/CHNzhwei/IE-MOIF/blob/master/IE-MOIF.png)
## Install
conda create -n IE-MOIF python=3.7<br>
conda activate IE-MOIF<br>
pip install -r requirements.txt<br>
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 –f https://download.pytorch.org/whl/torch_stable.html--user<br>
pip install ./utils_map/ lapjv-1.3.1.tar.gz<br>
## Usage
<b>Integration Step</b><br>
<<<<<<< HEAD
python IE-MOIF.py --data omics1_file omics_file omics3_file --label label_file --type omics_1_name omics_2_name omics_3_name --fs_num 1000 1000 500
<b>Classification Step</b><br>
python VIT.py --task num_class --patch num_patch --mark dataset
## Example

1. Ins python IE-MOIF.py --data E-MOIF.py ./dataset/PRAD/mRNA.csv ./dataset/PRAD/meth.csv ./dataset/PRAD/miRNA.csv --label ./dataset/PRAD/label.csv --type mRNA meth miRNA --drm fs --fs_num 1000 1000 500 --fem tsne<br>
2. python VIT.py --data ./dataset/PRAD/MoInter_output/5.MoInter_Transformed_Data_0.npy --label ./dataset/PRAD/MoInter_output/5.Data_label.npy --task 2 --mode 0 --patch 10 --mark PRAD<br>
=======
python main_IE-MOIF.py --data <i>omics1_file omics_file omics3_file</i> --label <i>label_file</i> --type <i>omics_1_name omics_2_name omics_3_name</i> --fs_num 1000 1000 500<br>
<b>Classification Step</b><br>
python main_VIT.py --task <i>num_class</i> --patch <i>num_patch</i> --mark <i>dataset</i>
## Example
1. python main_IE-MOIF.py --data E-MOIF.py ./dataset/PRAD/mRNA.csv ./dataset/PRAD/meth.csv ./dataset/PRAD/miRNA.csv --label ./dataset/PRAD/label.csv --type mRNA meth miRNA --drm fs --fs_num 1000 1000 500 --fem tsne<br>
2. python main_VIT.py --data ./dataset/PRAD/MoInter_output/5.MoInter_Transformed_Data_0.npy --label ./dataset/PRAD/MoInter_output/5.Data_label.npy --task 2 --mode 0 --patch 10 --mark PRAD<br>
>>>>>>> cee75beb1ed92716a51c329f59555d9953d1be09
