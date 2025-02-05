# Observations
- Models seem to underpredict score magnitude in top right coords.... why???

- VP SDE seems to be more stable than VE SDE

- 32 dim hidden features much faster and works fine for VP sde, experiment with more depth

- Noise conditional classifier seems to perform very poorly at any reasonable noise level... What helps?



# Needed:

- Visualisation for partial generation, i.e. along the SDE
    DONE!!!!!!!!!!!!!!!!!!!!!!!!

- More experiments at different depths, maybe increase dataset size?
    DONE!!!!!!!!!!!!!!!!!!!!!!!!

- Add classifier training
    DONE!!!!!!!!!!!!!!!!!!!!!!!!

- Compare noise conditional classifier against noise agnostic classifier on noisy data, see if it even matters


- Refactor to use TrainState more efficiently (no more mutils.get_model_fn...)