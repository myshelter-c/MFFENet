# MFFENet

# Preparation

The repository was built in PyTorch 1.7.1 and trained and tested on the environment (Python 3.7, CUDA 11.6).

Preparing the environment:    
  
```
conda create -n name python=3.7
conda activate name
```
Install PyTorch dependencies

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

# Datasets
Please refer to the data preparation method of [monodepth2](https://github.com/nianticlabs/monodepth2).

# Training
We provide training using [HRFormer pre-trained weights](https://drive.google.com/drive/folders/1IoVSDvF9zIPru1PvffuvtGx3XhtlLo7c?usp=drive_link). You can download it and place it in the models folder.  
Our model runs on two 16GB GPUs, and we support distributed training.
```
python -u -m torch.distributed.launch --nproc_per_node=2 train.py --model_name M_640x192_name --png --batch_size 8 --data_path data_path/kitti
```
# Testing
```
python evaluate_depth.py   --load_weights_folder models/M_640x192_name --eval_mono --eval_split eigen --data_path data_path/kitti
```
# Weights trained on the KITTI dataset
We provide weights for:
+ [MFFENet-tiny-model(640x192)](https://drive.google.com/file/d/1BegVLt9UbX1yld_lfBQYaQosejVV3wph/view?usp=drive_link)
+ [MFFENet-small-model(640x192)](https://drive.google.com/file/d/1EoC02VCdv3-TJXnTEGFezOZuQAktui6q/view?usp=drive_link)
# Acknowledgement
Thanks to the authors for their excellent work:
+ [HRFormer](https://github.com/HRNet/HRFormer)
+ [DIFFNet](https://github.com/brandleyzhou/DIFFNet)
+ [monodepth2](https://github.com/nianticlabs/monodepth2)
