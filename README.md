# DeepBoltzmannMachine

- dbm_wb.m- Basic implementation of a deep boltzmann machine
- sampling_wb.m- Sampling from the deep boltzmann machine

All models employ the early stopping criteria of stopping when the validation error does not decrease. This criteria can be relaxed or tightened in all model files by adjusting the parameter 'early_stopper_limit' which will allow the validation error to increase 'early_stopper_limit' times before stopping.

-All the parameter configurations can be adjusted at the beginning of the model files.

-Each model file also generates 3 files: 
	-The training and validation error trace in a csv file named with the details of the parameters
	-The best learned weights in the training round
	-The image of the visualized best learned weights before early stopping
