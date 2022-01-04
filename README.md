# Deep Active Learning with Contrastive Sampling

Deep Learning Project for Deep Learning Course (263-3210-00L)  
by Department of Computer Science, ETH Zurich, Autumn Semester 2021 

Authors:  
Sebastian Frey (sefrey@student.ethz.ch)  
Remo Kellenberger (remok@student.ethz.ch)  
Aron Schmied (aronsch@student.ethz.ch)  
Guney Tombak (gtombak@student.ethz.ch)  

Professors:  
Dr. Fernando Perez-Cruz   
Dr. Aurelien Lucchi

## Setup

All dependencies can be fulfilled by creating a conda environment using `environment.yaml`:  

```shell
conda env create -f environment.yaml && conda activate dalcs
```

For GPU implementation use also:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=X.Y -c pytorch -c conda-forge
```

with a cuda version X.Y compatible with your GPU.

Before running, you should login to your wandb (Weights and Biases) account:

```shell
wandb login
```

You can change the parameters regarding Weights and Biases in `main.py` at line `70`.

For more information, please visit [Weights and Biases](https://wandb.ai).

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

## Additional

### Results by Weights and Biases

The results are saved in both your local device in the folder named `wandb` and also Weights and Biases cloud. You can inspect the results directly on web or to use the local files, please check the [documentation](https://docs.wandb.ai/guides/track/public-api-guide) and code `visualization/plot_results.ipynb`.

### Variational Autoencoder Training
The code also contains a seperate variational autoencoder trainer to use pretrained models.

To use it, set your current directory to `vae_training`. Usage is similar to the main file:

```shell
python train_vae.py --config <configuration_file_path>
```

