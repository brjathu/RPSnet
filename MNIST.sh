#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python3 mnist.py 0 0 & 
CUDA_VISIBLE_DEVICES=1 python3 mnist.py 1 0 & 
CUDA_VISIBLE_DEVICES=2 python3 mnist.py 2 0 & 
CUDA_VISIBLE_DEVICES=3 python3 mnist.py 3 0 & 
CUDA_VISIBLE_DEVICES=4 python3 mnist.py 4 0 & 
CUDA_VISIBLE_DEVICES=5 python3 mnist.py 5 0 & 
CUDA_VISIBLE_DEVICES=6 python3 mnist.py 6 0 & 
CUDA_VISIBLE_DEVICES=7 python3 mnist.py 7 0


sleep 5
CUDA_VISIBLE_DEVICES=0 python3 mnist.py 0 1 & 
CUDA_VISIBLE_DEVICES=1 python3 mnist.py 1 1 & 
CUDA_VISIBLE_DEVICES=2 python3 mnist.py 2 1 & 
CUDA_VISIBLE_DEVICES=3 python3 mnist.py 3 1 & 
CUDA_VISIBLE_DEVICES=4 python3 mnist.py 4 1 & 
CUDA_VISIBLE_DEVICES=5 python3 mnist.py 5 1 & 
CUDA_VISIBLE_DEVICES=6 python3 mnist.py 6 1 & 
CUDA_VISIBLE_DEVICES=7 python3 mnist.py 7 1


sleep 5
CUDA_VISIBLE_DEVICES=0 python3 mnist.py 0 2 & 
CUDA_VISIBLE_DEVICES=1 python3 mnist.py 1 2 & 
CUDA_VISIBLE_DEVICES=2 python3 mnist.py 2 2 & 
CUDA_VISIBLE_DEVICES=3 python3 mnist.py 3 2 & 
CUDA_VISIBLE_DEVICES=4 python3 mnist.py 4 2 & 
CUDA_VISIBLE_DEVICES=5 python3 mnist.py 5 2 & 
CUDA_VISIBLE_DEVICES=6 python3 mnist.py 6 2 & 
CUDA_VISIBLE_DEVICES=7 python3 mnist.py 7 2


sleep 5
CUDA_VISIBLE_DEVICES=0 python3 mnist.py 0 3 & 
CUDA_VISIBLE_DEVICES=1 python3 mnist.py 1 3 & 
CUDA_VISIBLE_DEVICES=2 python3 mnist.py 2 3 & 
CUDA_VISIBLE_DEVICES=3 python3 mnist.py 3 3 & 
CUDA_VISIBLE_DEVICES=4 python3 mnist.py 4 3 & 
CUDA_VISIBLE_DEVICES=5 python3 mnist.py 5 3 & 
CUDA_VISIBLE_DEVICES=6 python3 mnist.py 6 3 & 
CUDA_VISIBLE_DEVICES=7 python3 mnist.py 7 3


sleep 5
CUDA_VISIBLE_DEVICES=0 python3 mnist.py 0 4 & 
CUDA_VISIBLE_DEVICES=1 python3 mnist.py 1 4 & 
CUDA_VISIBLE_DEVICES=2 python3 mnist.py 2 4 & 
CUDA_VISIBLE_DEVICES=3 python3 mnist.py 3 4 & 
CUDA_VISIBLE_DEVICES=4 python3 mnist.py 4 4 & 
CUDA_VISIBLE_DEVICES=5 python3 mnist.py 5 4 & 
CUDA_VISIBLE_DEVICES=6 python3 mnist.py 6 4 &
CUDA_VISIBLE_DEVICES=7 python3 mnist.py 7 4

