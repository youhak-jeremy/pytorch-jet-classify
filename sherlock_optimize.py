import itertools
import numpy as np
from sherlock import Sherlock
from typing import List
import models
from iter_quantize import execute, load_jet_data
from tools.parse_yaml_config import parse_config

def step(X: np.ndarray, y: np.ndarray, known_idx: List[int], dataloaders):
    print(known_idx)

    for idx in known_idx:
        performance, efficiency = execute(
            dataloaders,
            output_dir="./train_output",
            num_epochs=10,
            quantization_spec=X[idx, :].tolist()
        ) 
        
        y[known_idx, :] = [1 / performance, efficiency] # Minimize the loss, Maxmize the efficiency

if __name__ == '__main__':
    # For the inputs, outputs of the model
    yaml_config = parse_config("./configs/train_config_threelayer.yml")
    train_file="./train_data/train"
    test_file="./train_data/test"
    dataloaders = load_jet_data(train_file, test_file, yaml_config)

    # QUESTION: WHY 5??!!
    layer_1_quantization = [32, 12, 8, 6, 4]
    layer_2_quantization = [32, 12, 8, 6, 4]
    layer_3_quantization = [32, 12, 8, 6, 4]
    layer_4_quantization = [32, 12, 8, 6, 4]
    layer_5_quantization = [32, 12, 8, 6, 4]

    X = np.array(list(itertools.product(
            layer_1_quantization,
            layer_2_quantization,
            layer_3_quantization,
            layer_4_quantization,
            layer_5_quantization,
        )))
    
    count, _ = X.shape
    y = np.zeros([count, 2])
    
    percent = 0.3
    budget = int(round(percent * count))

    sherlock = Sherlock(
        n_init=5,
        budget=budget,
        surrogate_type="rbfthin_plate-rbf_multiquadric-randomforest-gpy",
        kernel="matern",
        num_restarts=0,
        pareto_margin=0,
        y_hint=None,
        plot_design_space=False,
        use_ted_in_loop=True,
        request_output=lambda y, idx: step(
            X,
            y,
            idx,
            dataloaders,
        ),
        action_only=None,
        n_hint_init=0,
        scale_output=True,
        use_trace_as_prior=True,
        model_selection_type="mab10",
    )
    
    sherlock.fit(X).predict(X, y)