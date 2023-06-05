# Sherlock-Q 


## Overview 
This is our code repository for CSE 237D Spring 2023 Project -- Sherlock Q. 

Our Sherlock-Q project is about developing an Auto-Quantizing Hardware Neutral system, which aims to solve some challenges faced in deploying deep learning models, specifically in computation and memory requirements. The "Q" stands for "Quantizing," highlighting the project’s main objective.
Sherlock-Q is a powerful solution for automating the quantization process,which allows users to efficiently optimize their neural networks for various hardware platforms.
For instance, ChatGPT incurs a significant daily cost (~$700K or more) and utilizes around 30K GPUs.


## Abstract:
Neural network quantization, critical for efficient network deployment on hardware like FPGAs and ASICs, often requires expert intervention and significant computational resources. This project automated this process using Sherlock-Q, a system developed from the Sherlock design space exploration framework. Sherlock-Q automates hardware quantization of neural networks, minimizing manual intervention and hardware design cycle time. Consequently, it allows more efficient design space exploration for network quantization, reducing the need for hardware experts and computational resources.

## Team Members:
- Goldar Luu
- Kai Yui Yeung
- Vineet Thumuluri
- You Lee


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

This will generate two folders:

 1. “checkpoints” which contains the trained models
 2. “metrics” with json files for each trained model containing metrics related to the performance, model size, training curves, etc.


Use the “plot_pareto.ipynb” notebook to generate a plot like the one below.
![alt text](images/pareto_front.png)


The graph above illustrates an example choice which results in approximately 80% fewer computations, yet causes less than a 3% drop in accuracy.








