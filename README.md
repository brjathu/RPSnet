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

for ILSVRC, download the pickle file contains file names and labels. Place it inside `prepare` and run `python3 prepare/generate_ILSVRC_labels.py`


## Performance




## Download pre-trained models

Download this [CIFAR10 - pretrained models](https://drive.google.com/open?id=1Plj-dH4OoSORqWf-23XxToW0X46NdVmR) and extract the files inside model directory. Then run `ensemble.py` file.
```
python ensemble.py
```

## We credit
We have used [this](https://github.com/XifengGuo/CapsNet-Keras) as the base CapsNet implementation. We thank and credit the contributors of this repository.

## Contact
Jathushan Rajasegaran - brjathu@gmail.com  
Discussions, suggestions and questions are welcome!

## References
[1] J. Rajasegaran, V. Jayasundara, S.Jeyasekara, N. Jeyasekara, S. Seneviratne, R. Rodrigo. "DeepCaps : Going Deeper with Capsule Networks." *Conference on Computer Vision and Pattern Recognition.* 2019. [[arxiv]](https://arxiv.org/abs/1904.09546)


---

If you found this code useful in your research, please consider citing
```
@misc{rajasegaran2019deepcaps,
    title={DeepCaps: Going Deeper with Capsule Networks},
    author={Jathushan Rajasegaran and Vinoj Jayasundara and Sandaru Jayasekara and Hirunima Jayasekara and Suranga Seneviratne and Ranga Rodrigo},
    year={2019},
    eprint={1904.09546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
