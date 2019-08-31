# Set-processing

The implementation and code for the paper:  

L. Maziarka, M. Smieja, A. Nowak, J. Tabor, L. Struski, P. Spurek, ***Set aggregation network for structured data processing***,
arXiv preprint: https://arxiv.org/abs/1810.01868, 2018
  
  
The reposiory consists of 
  - README.md - this file
  - src - the main code directory, divided into:
    - data handling - the function helpers for loading and preprocessing the data
    - layer - the SAN implementation
    - metrics - code for gathering and reporting the results
    - model - the implementation of the used in experiments models
    - train - all train related logic and main programs.
    
## Running the code:

An example of running the code for small-scale convolutional network for the CIFAR-10 experiment: 

`python src/train/image_main.py --folder path/to/output/folder --head-hidden-dim 256 --head-layers 1 --shuffle --print-params --l2 0.0 --batch-size 256 --learning-rate 1e-3 --epochs 15  --body-type cnn-avg --dropout 0.0`

___

Where:

- `--folder` - the path to the folder with outputs
- `--head-hidden-dim` - the hidden dimension of the predictor head
- `--head-layers` - number of hidden layers in the predictor head
- `--shuffle` - shuffle the input dataset
- `--print-params` - print the passed command line arguments
- `--l2` - the used l2 regularization scale
- `--batch-size` - the size of one batch
- `--learning-rate` - the used learning rate
- `--epochs` - the number of epochs
- `--body-type` - the type of used body model and the corresponding aggregation method. (see `--help`)
- `--dropout` - the dropout rate

For more options and their description please run:

`python src/train/image_main.py --help`

Please note that the default values of hyper-parameters in the code are not necessarily equal to the values used in the paper. For the hyper-parameter setting please refer to the experiments description in the paper. 

____



