# Deep Learning Project

Authors: Sebastian Frey, Remo Kellenberger, Aron Schmied, and Guney Tombak  
Deep Learning, Department of Computer Science, ETH Zurich  

## Setup

All dependencies can be fulfilled by creating a conda environment using environment.yml  

```shell
conda env create -f environment.yml && conda activate dlp
```

For GPU implementation use also:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=X.Y -c pytorch -c conda-forge
```

## Usage

The parameters of the run can be configured by using the `config.py`:

```shell
python main.py
```

The results are saved using the Weights and Biases system.

