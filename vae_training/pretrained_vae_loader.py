"""loads pretrained models from polybox"""
import requests
import zipfile
import io
import os

def download_vae():

    path = 'vae_training/pretrained_models'

    links = [
        # r'https://polybox.ethz.ch/index.php/s/qDeyUawrrICvKxJ/download',
        r'https://polybox.ethz.ch/index.php/s/G15Y7HXicHv0sYq/download',
        # r'https://polybox.ethz.ch/index.php/s/oU3svRBRW6TWItZ/download',
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