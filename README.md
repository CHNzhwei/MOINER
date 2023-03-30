# IE-MOIF: a multi-omics integration framework with information enhancement and image representation learning
## IE-MOIF Introduction
![image](https://github.com/CHNzhwei/IE-MOIF/blob/master/IE-MOIF.png)
## Install
```bash
conda create -n IE-MOIF python=3.7
conda activate IE-MOIF
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 –f https://download.pytorch.org/whl/torch_stable.html --user
pip install ./utils_map/ lapjv-1.3.1.tar.gz
```
## Usage
### Integration Step
```bash
python main_IE-MOIF.py --data omics1_file omics2_file omics3_file --label label_file --type omics_1_name omics_2_name omics_3_name --fs_num 1000 1000 500
```
### Classification Step

```bash
python VIT.py --task num_class --patch num_patch --mark dataset
```
## Example

```bash
python main_IE-MOIF.py --data ./example/mRNA.csv ./example/meth.csv ./example/miRNA.csv --label ./example/label.csv --type mRNA meth miRNA --drm fs --fs_num 1000 1000 500 --fem tsne<br>
python main_En-VIT.py --n_class 2 --patch 25 --Example
```
## NOTE
In the IE-MOIF framework, ViT is used as the default classification model, and En-ViT based on ensemble learning needs to be manually set to open in parameters because it takes a long time to train.




