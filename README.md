# RPSnet
Official Implementation of "Random Path Selection for Incremental Learning" paper. 

This code provides implementation for RPSnet : Random Path Selection network for incremental learning. This repository is implemeted with pytorch and the scripts are written to run the experiments on a DGX machine.

## Usage
### step 1 : Install dependencies
```
conda install -c pytorch pytorch
conda install -c pytorch torchvision 
conda install -c conda-forge matplotlib
conda install -c conda-forge pillow
```
### step 2 : Clone the repository
```
git clone https://github.com/brjathu/RPSnet.git
cd RPSnet
```

## Supported Datasets
 - `CIFAR100`
 - `ILSVRC`
 - `CIFAR10` 
 - `SVHN` 
 - `MNIST`

 
## Training

The algorithm requires `N` GPUs in parallel for random path selection. The script uses `N=8` with a DGX machine.  
```
./CIFAR100.sh
```

If you are training on single GPU change the `CIFAR100.sh` as well as other running scripts to use only one GPU (comment out the rest).

If you are training on for a specific task, then run,
```
python3 cifar.py TASK_NUM TEST_CASE
```

To test with several other datasets (ILSVRC, MNSIT etc) run the corresponding files `imagenet.py` or `mnist.py`.

If you need to change the settings (Eg. Number of tasks), run the `python3 prepare/generate_cifar100_labels.py` with corresponding settings.

for ILSVRC, download the pickle file  [meta_ILSVRC.pkl](https://drive.google.com/open?id=1Plj-dH4OoSORqWf-23XxToW0X46NdVmR) contains file names and labels. Place it inside `prepare` and run `python3 prepare/generate_ILSVRC_labels.py`


## Performance


![a](https://drive.google.com/open?id=13lAKnkgAYJOR4IAWFBwH2c79yZpl3FtB?raw=True)

## Download pre-trained models

Download this [model](https://drive.google.com/file/d/1VFxDq6CrAaIeQda_JWVb01erz8BJnlZk/view?usp=sharing) and extract the files inside models directory. Then run `python3 test.py` file with corresponding paths.



## We credit
We have used [this](https://github.com/kimhc6028/pathnet-pytorch) as the pathnet implementation. We thank and credit the contributors of this repository.

## Contact
Jathushan Rajasegaran - jathushan.rajasegaran@inceptioniai.org   or brjathu@gmail.com
Discussions, suggestions and questions are welcome!


