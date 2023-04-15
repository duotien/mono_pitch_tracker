# Mono Pitch Tracker | <a href="https://colab.research.google.com/github/duotien/mono_pitch_tracker/blob/main/notebooks/mono_pitch_tracker.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height=30/></a>

A data driven Mono Pitch Tracker using Pytorch.

# Motivation

This project is a companion code for our undergraduate thesis. The aim of this project is to:
- Research and apply our knowledge to solve a Melody Extraction problem using ML techniques and algorithms.
- Apply our ML model to create a basic monophonic pitch tracker.
- Learn something new.

# Setup
We recommend using anaconda/mini-conda to setup the project.

The project aslo contains a forked version of [medleydb](https://github.com/marl/medleydb).

1. Clone this repo and its submodules:
```
git clone --recursive https://github.com/duotien/mono_pitch_tracker.git
```

2. Create a conda environment:
```
cd mono_pitch_tracker
conda env create -f environment.yml
conda activate mpt_2022
```

3. Use conda-develop to install the package to the environment (this help importing custom modules)

Note: you must still be in the mono_pitch_tracker directory:

```
conda develop .
```

4. Do the same thing for medleydb:
```
cd medleydb
conda develop .
```

5. Follow this guide to install `pytorch` according to your device: https://pytorch.org/get-started/locally/

# Usage
An example usage for this project is making a simple monophonic Auto Music Transcriptor. We provide a script to let user extract the melody from an audio input and convert it into midi using `mido`.

```
python audio2midi.py audio.mp3 -o melody.mid
```

# Reproduce
If you would like to reproduce this experiment, start with the jupyter notebooks in `notebooks/`.

## Dataset
Below are the datasets used in this experiment:

- [MIREX05 & ADC2004](http://labrosa.ee.columbia.edu/projects/melody/) (for testing)
- [MedleyDB](https://medleydb.weebly.com/) [1] (for training & testing)

The MedleyDB audio files are available through permission request. If you want to see how the dataset is structured, you can download 2 samples on their [download webpage](https://medleydb.weebly.com/downloads.html).

The data splits for MedleyDB is located in `./pitch_tracker/utils/data_split.json`. We only keep the test split for comparison. Train & Validation splits are merged and resplitted by *genres* in `5_data_loader.ipynb`.

## Model

Our melody extraction model is based on [basaran2018CRNN](https://github.com/dogacbasaran/ismir2018_dominant_melody_estimation/blob/master/CRNN/C-RNN_model1.py) [2]. However, we use mel-spectrogram as our model input and output 88 notes based on the 88 keys piano.

# Citations & Acknowledgements
> [1] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, Oct. 2014. \
> [2] D. Basaran, S. Essid and G. Peeters, Main Melody Extraction with Source-Filter NMF and CRNN, In 18th International Society for Music Information Retrieval Conference, ISMIR, 2018, Paris, France
