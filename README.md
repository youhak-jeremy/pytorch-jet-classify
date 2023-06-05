# Sherlock-Q 


## Overview 
This is our code repository for CSE 237D Spring 2023 Project -- Sherlock Q. 

Our Sherlock-Q project is about developing an Auto-Quantizing Hardware Neutral system, which
aims to solve some challenges faced in deploying deep learning models, specifically in computation
and memory requirements. The "Q" stands for "Quantizing," highlighting the project’s main objective.
Sherlock-Q is a powerful solution for automating the quantization process,which allows users to
efficiently optimize their neural networks for various hardware platforms.

## Abstract:
Neural network quantization is a complex process that typically demands expert intervention and significant computational resources. This complexity forms a barrier to the efficient deployment of neural networks, particularly in hardware environments like FPGAs and ASICs where resources are finite and precision is paramount. This project was undertaken to automate the design space exploration for hardware quantization of neural networks, a process vital for optimizing the performance of deep learning models on resource-constrained devices.We used Sherlock, a design space exploration framework, to develop Sherlock-Q, an Auto-Quantizing Hardware Neutral system. This system aims to automate the quantization process, thus removing the need for manual intervention and significantly reducing hardware design cycle time. The implementation of Sherlock-Q successfully automated the process of hardware quantization for neural networks. This allowed us to carry out design space exploration for neural network quantization more efficiently, significantly reducing the involvement of hardware experts and decreasing the required computational resources.

## Team Members:
- Goldar Luu
- Kai Yui Yeung
- Vineet Thumuluri
- You Lee
- 
## Requirements:
- Python 3.8
- To use this codebase to determine the performance of Sherlock-Q, one must satisfy all requirements. It is recommend that one sets up in an virtual environment to avoid conflicts. All the libraries used are listed in ```requirements.txt```. Run the following command to instlal all required libraries. 

```
pip install -r requirements.txt
```

## Usage: 

Train the 3-layer JetTagger model using the below command

```
python sherlock_optimize.py
```
This will generate two folders, “checkpoints” and “metrics” 

Use the “plot_pareto.ipynb” notebook to generate the plot
![alt text](https://github.com/youhak-jeremy/pytorch-jet-classify/blob/master/plot_pareto_front.png?raw=true)









