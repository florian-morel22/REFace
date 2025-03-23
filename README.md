# REFace 

This repo is a fork of the original repo : https://github.com/Sanoojan/REFace.git

As part of a project on deepfake generation and detection, I worked on understanding and adapting the REFace model. The majority of my work can be found in the /our_work folder at the root of the project.

The goal of the global project is the generation of a deepfake dataset through different technologies, Diffusion, GAN and VQVAE in order to train a deepfake detector. You can find the global repository heare : https://github.com/florian-morel22/deepfake-project.git

# Run the code

First, to install the necessary packages and models for the code to function properly, please run:
```
make setup
```

## Simple generation

In the Makefile, enter the paths of the images to be face-swapped after ```SOURCE=``` and ```TARGET=```.

Then, run the following command :

```
make faceswapp
```
or, if you prefer head-swapping :
```
make headswapp
```

## Dataset generation

Download the CelabA dataset with the following command :

```
make download-celeba
```

Then, enter the name of the huggingface dataset you want to create after ```HF_IDENTIFIER_DATASET=```. Enter your HuggingFace token in a ```.env``` file as following :

```.env
HF_TOKEN=...
```

and run :

```
make generate-dataset
```

If the HuggingFace dataset already exists, it will add the newly generated images, avoiding duplicates.