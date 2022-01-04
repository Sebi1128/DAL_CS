"""
Deep Active Learning with Contrastive Sampling

Deep Learning Project for Deep Learning Course (263-3210-00L)  
by Department of Computer Science, ETH Zurich, Autumn Semester 2021 

Authors:  
Sebastian Frey (sefrey@student.ethz.ch)  
Remo Kellenberger (remok@student.ethz.ch)  
Aron Schmied (aronsch@student.ethz.ch)  
Guney Tombak (gtombak@student.ethz.ch)  
"""

import requests
import zipfile
import io
import os

def download_vae():
    """
    Downloading pretrained variational autoencoder parameters
    """

    path = 'vae_training/pretrained_models'

    links = [
        r'https://polybox.ethz.ch/index.php/s/qDeyUawrrICvKxJ/download',
        r'https://polybox.ethz.ch/index.php/s/G15Y7HXicHv0sYq/download',
        r'https://polybox.ethz.ch/index.php/s/oU3svRBRW6TWItZ/download',
        r'https://polybox.ethz.ch/index.php/s/PATJkvh3VQX1n5s/download'
    ]

    if not os.path.exists(path):
        os.makedirs(path)

    for link in links:
        print("load ", link)
        r = requests.get(link)
        print("unpack ", link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)

if __name__ == "__main__":
    download_vae()