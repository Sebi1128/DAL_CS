# Deep Learning Project

Authors: Sebastian Frey, Remo Kellenberger, Aron Schmied, and Guney Tombak  
Deep Learning, Department of Computer Science, ETH Zurich  

## Setup

All dependencies can be fulfilled by creating a conda environment using environment.yaml:  

```shell
conda env create -f environment.yaml && conda activate dalcs
```

For GPU implementation use also:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=X.Y -c pytorch -c conda-forge
```

with a cuda version X.Y compatible with your GPU.

## Usage

The parameters of the run can be configured using YAML files. The predefined configurations can be found in `configs` folder.

```shell
python main.py --config <configuration_file_path>
```

To run more than one configuration, `multi_main.sh` can be used with the path of the configuration files to be sequentially run.

```shell
conda activate dalcs
chmod +x multi_main.sh
./multi_main.sh -c <folder_path_containing_configuration_files>
```

An explanatory configuration file can be found at `configs/default_config.yaml`.

## Results

The results are saved using the Weights and Biases system.

## Additional

### Variational Autoencoder Training
The code also contains a seperate variational autoencoder trainer to use pretrained models.

To use it, set your current directory to `vae_training`. Usage is similar to the main file:

```shell
python train_vae.py --config <configuration_file_path>
```

