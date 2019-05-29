# RPSnet
Official Implementation of "Random Path Selection for Incremental Learning" paper. 

This code provides an implementation for RPSnet : Random Path Selection network for incremental learning. This repository is implemented with pytorch and the scripts are written to run the experiments on a DGX machine.

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



## Introduction

<p align="justify">Incremental life-long learning is the main challenge towards the long-standing goal of Artificial General Intelligence. In real-life settings, learning tasks arrive in a sequence and machine learning models must continually learn to increment already acquired knowledge. Existing incremental learning approaches, fall well below the state-of-the-art cumulative models that use all training classes at once. In this paper, we propose a random path selection algorithm, called RPS-Net, that progressively chooses optimal paths for the new tasks while encouraging parameter sharing and reuse. Our approach avoids the overhead introduced by computationally expensive evolutionary and reinforcement learning based path selection strategies while achieving considerable performance gains. As an added novelty, the proposed model integrates knowledge distillation and retrospection along with the path selection strategy to overcome catastrophic forgetting. In order to maintain an equilibrium between previous and newly acquired knowledge, we propose a simple controller to dynamically balance the model plasticity.  Through extensive experiments, we demonstrate that the proposed method surpasses the state-of-the-art performance on incremental learning and by utilizing parallel computation this method can run in constant time with nearly the same efficiency as a conventional deep convolutional neural network.</p>


<p align="center"><img src="https://drive.google.com/uc?export=view&id=13lAKnkgAYJOR4IAWFBwH2c79yZpl3FtB"></p>
<p align="center">(a) RPS network architecture</p>
<br/>
<br/> 
<p align="center"><img src="https://drive.google.com/uc?export=view&id=1dGBisyVzJpIRxc1BP9v_DU5-wgMuzMuy"></p>
<p align="center">(b) Random path selection algorithm</p>


## Performance


<p align="center"><img src="https://drive.google.com/uc?export=view&id=1xxjd_DVWTZ-jXauD_IduPtNTNZqum1I_" width="500"></p>
<p align="center">(c) Performance on CIFAR100 with 10 classes per task</p>
<br/>
<br/>
<p align="center"><img src="https://drive.google.com/uc?export=view&id=1uYxRTszoIi1UaM8y4_njhXevDTr8KgBy" width="500"></p>
<p align="center">(d) Performance on CIFAR100 with 20 classes per task</p>
<br/>
<br/>
<p align="center"><img src="https://drive.google.com/uc?export=view&id=1-_5o3OixdpfVqglQS3xLdDda5H1yuCj0" width="500"></p>
<p align="center">(e) Performance on CIFAR100 with 50 classes per task</p>
<br/>
<br/>
<p align="center"><img src="https://drive.google.com/uc?export=view&id=1ZihpdFfGUtZDGrgA-ul0JA7RNCjvTVXf" width="500"></p>
<p align="center">(f) Performance on Imagenet 100 classes with 10 classes per task</p>





## We credit
We have used [this](https://github.com/kimhc6028/pathnet-pytorch) as the pathnet implementation. We thank and credit the contributors of this repository.

## Contact
Jathushan Rajasegaran - jathushan.rajasegaran@inceptioniai.org   or brjathu@gmail.com
<br/>
Discussions, suggestions and questions are welcome!


