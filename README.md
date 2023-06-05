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

## Training:

```
python ./iter_prune.py -i train_data/train/ -t train_data/test/ -c ./configs/train_config_threelayer.yml -e 250 -o ./train_output/ -L -m 32,12,6,4 -n -s
```

```
python3 train.py -i <input H5 File/Directory of Files to train on> -o <output directory> -t <H5 File/Directory of files to use as test set> -e <num epoch to train for> -c <config to use>
```

## Generating Metrics 
```
python ./sherlock_optimize.py
```

Upon training completion, graphs for the ROC AUC vs Epoch, Loss vs Epoch, Precision vs Epoch, ROC for each tagger, and Confusion Matrix are saved to the output directory, along with a .pt saved model file. 

The Float/Unquantized 3 Layer model is `models.three_layer_model()`, with the Quantized/Brevitas 3 Layer Model is `models.three_layer_model_bv()`. Either can be chosen to train by setting `current_model` to one of the two. 

At the moment, the precision of `models.three_layer_model_bv()` is set by `self.weight_precision` within the class in `models.py`, though this is likely to change in the future

## PRP Nautilus Kubernetes cluster (https://nautilus.optiputer.net/) instructions:

First create the persistent volume claim (PVC):
```
kubectl create -f pt-jet-class-vol.yml
```
This is used to store the data and model outputs so they persist after deleting pods and jobs.

To do interactive work:
```
# create the pod
kubectl create -f pt-jet-class-pod.yml
# login to the pod
kubectl exec -it pt-jet-class-pod bash
```

In particular, you can populate the PVC with the data:
```
cd /ptjetclassvol/
mkdir data
wget https://raw.githubusercontent.com/ben-hawks/pytorch-jet-classify/master/jet_data_download.sh
source jet_data_download.sh
```

To check on running pods:
```
kubectl	get pods
kubectl	describe pods pt-jet-class-pod
```

To delete the pod:
```
kubectl delete pods pt-jet-class-pod
```
It also auto-deletes after 6 hours.

To launch a job:
```
kubectl create -f pt-jet-class-job.yml
```

To check on running jobs:
```
kubectl get jobs
```

You can also get the logs of the running jobs by getting the pod name first through

```
# get job's pod name
kubectl get pods
kubectl describe jobs pt-jet-class-job
# with pod's name, get logs
kubectl logs pt-jet-class-job-baseline-<random-string>
```





