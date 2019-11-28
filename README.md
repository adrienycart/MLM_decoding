# MLM_decoding

In this branch, we make available the code to reproduce the experiments from the following publication

> Adrien Ycart, Andrew McLeod, Emmanouil Benetos and Kazuyoshi Yoshii. "Blending Acoustic and Language Model Predictions for Automatic Music Transcription", _20th International Society for Music Information Retrieval Conference (ISMIR)_, November 2019, Delft, Netherlands.

If you use any of this in your works, please cite:

```  
    @inproceedings{ycartmcleod2019, 
       Address = {Delft, Netherlands},     
       Author = {Ycart, Adrien and McLeod, Andrew and Benetos, Emmanouil and Yoshii, Kazuyoshi},    
       Booktitle = {20th International Society for Music Information Retrieval Conference},    
       Month = {Nov.},    
       Title = {Blending Acoustic and Language Model Predictions for Automatic Music Transcription},    
       Year = {2019}    
    }  
```  

Additional material (figures, sound examples...) can be found on this webpage: [http://c4dm.eecs.qmul.ac.uk/ycart/ismir19.html](http://c4dm.eecs.qmul.ac.uk/ycart/ismir19.html)

## Installation
The easiest way to install is using an environment manager like conda. We have tested the code with tensorflow 1.11, but other versions may work as well. To install, run the following commands:

```
conda create -n blending python=3.6 tensorflow=1.11 numpy=1.15.4
conda activate blending
pip install -r requirements.txt
```

## Using Pre-trained Models
We provide in this repository everything needed to reproduce the results from our experiments, that includes:

* [`data`](./data): MIDI files and acoustic model outputs as csvs, split into test, train, and validation sets.
* [`MLMs`](./MLMs): Pre-trained Music Language Models, for 40ms and 16th note (quant) timesteps, both with and without scheduled sampling.
* [`blending_model/models`](./blending_model/models): Pre-trained weight and prior models for each MLM.

Instructions for creating the data and re-training the models can be found in [Starting from Scratch](#starting-from-scratch).

Evaluate the model with the following command:
```
python evaluate.py model data [--step {time,quant}] [-wm weight_model | -w FLOAT] [--save output_path]
```

* `model`: A saved MLM, without its file extension.
* `data`: A data directory, usually `data/test`
* `--step {time,quant}`: `time` (default) to use 40ms timesteps, `quant` to use 16th-note timesteps.
* `-wm weight_model`: The weight or prior model to use during testing.
* `-w FLOAT`: The weight to use for a constant weight model. Use `-w 1.0` to use the acoustic model only, thresholding at 0.5.
* `--save output_path`: Save results in the given directory.

For example, to reproduce the results of the prior model with scheduled sampling and 16th-note timesteps, use the following command:
```
python evaluate.py MLMs/scheduled/quant/best_model.ckpt-1194 data/test --step quant -wm blending_model/models/pm.quant-sched.pkl
```

**Important**: The results with 16th-note timesteps will be identical to those in the paper. For the 40ms timestep, we use an additional post-processing step to remove short gaps. To do this, run evaluate.py with `--save output_path`, and then evaluate with: `python evaluate_load.py output_path data/test --gap`

## Starting from Scratch

### Step 1: Prepare the data

You need the following data for these experiments:

- MAPS dataset: http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/
- A-MAPS annotations (for 16th note positions): http://c4dm.eecs.qmul.ac.uk/ycart/a-maps.html
- Piano-midi.de MIDI files (to train the Music Language Model): http://piano-midi.de/

#### Acoustic model

You will then need to compute the acoustic model outputs.
To do so, follow the instructions from this repository: https://github.com/rainerkelz/framewise_2016
We used the following split: [Acoustic model split](http://c4dm.eecs.qmul.ac.uk/datasets/ycart/ismir19/split_MAPS.zip)

Once the outputs have been computed, place the CSV files in a folder, along with the corresponding A-MAPS MIDI files.
Every MIDI file should have one CSV file, name identically, with only the extension differing (<file>.csv and <file>.mid).

Our aligned CSV and MIDI data is provided in [data](./data).

#### Music Language model (MLM)

You should split the Piano-midi.de files similarly to the split used for the acoustic model, meaning that all the pieces in the MAPS test set are used for testing, all the pieces in the MAPS validation set are used for validation, and all the remaining pieces are used for training.
We used the following split: [MLM split](http://c4dm.eecs.qmul.ac.uk/datasets/ycart/ismir19/split_PM.zip)

You should then put the MIDI files inside a folder containing 3 subfolders called ``train``, ``valid`` and ``test``, and containing the training, validation and testing MIDI files, respectively.
http://c4dm.eecs.qmul.ac.uk/datasets/ycart/ismir19/split_MAPS.zip

### Step 2: Train the MLM

First, you need to create a folder named ``ckpt``.
Any save path you then use will automatically create a folder inside ``ckpt`` (no need to specify the ``ckpt/`` prefix).

To train the model, use ``python train_mlm.py``.
You can show arguments and options with ``python train_mlm.py -h``.
In our experiments, we used default parameters.
- To use 16th note timesteps, use ``-quant`` (otherwise: use 40ms timesteps)
- To train the MLM with scheduled sampling, use ``-sched_sampl self``

At this stage, you can already evaluate your model using a fixed weight (see Step 4: Evaluate the model).

Our pre-trained MLMs are in [MLMs](./MLMs).

### Step 3: Train the blending model

#### Create training data

First you need to create training data for the blending model.
The data should be created using the validation files.
To do so, use ``python blending_model/optim/create_weight_data.py data/valid -m MODEL --out out_file [ARGS]``
Here, `MODEL` is a pointer to a trained MLM checkpoint, without any extension (like `MLMs/scheduled/quant/best_model.ckpt-1194`).

Additional args can be displayed with ``-h``
We used all default parameters except:
* ``--step quant`` for 16th-note timesteps.
* ``--min_diff`` is set to 0.0 for 16th note steps, and to 0.1 with 40ms timesteps (otherwise, the created data is too big)
* ``--hist`` is set to 10 for 16th note steps, and 50 for 40ms timesteps, to capture a comparable time window in both cases.
* ``--features``: include to use handcrafted features (we did in our experiments)

NOTE: This step takes some time (20-30 minutes with 40ms timesteps), but it works.

We do not provide this data as it is quite large (>100MB each), and easy to generate.

#### Run Bayesian Optimisation

Then, you need to run the Bayesian Optimisation.
To do so, use ``python blending_model/optim/optimize_sk.py model data [ARGS]``.
`model` is a pointer to a trained MLM checkpoint, without any extension (like `MLMs/scheduled/quant/best_model.ckpt-1194`).  
`data` should be a pointer to the validation data (`data/valid`).

Other options can be displayed with ``-h``.
Important options are:
* ``--step quant``: for 16th-note timesteps.
- ``--beam_data``: should point to the file created in the previous step
- ``--prior``: use this to train a Prior Model (otherwise: train a Weight Model)
- ``--model_dir``: point to a specific location where all the trained blending models will be saved


**IMPORTANT**: Save the output of this step!! For instance: ``python blending_model/optim/optimize_sk.py ARGS > out.txt ``
You need that to be able to retrieve the best performing model at the end of the Bayesian Optimisation process.

To get the best weight model, use ``grep "^0." out.txt | sort -n``
(where out.txt is the output of the previous step).
The last line should be of the form : ``<notewise_f_measure> : [hyperparameters of the corresponding model]``

For instance, the last line could be: ``0.8377122624376068: [False, 0.6336185697575645, 3, 3, False, True, 0, 0, True]``.
This would correspond to a model named ``weight_model.b10_md0.6336185697575645_h3_l3_f_hc0_pc0_prior.quant.0.pkl`` if ``--prior`` and ``--step quant`` were used.

### Step 4: Evaluate the model

Evaluate the model with the following command:
```
python evaluate.py model data [--step {time,quant}] [-wm weight_model | -w FLOAT] [--save output_path]
```

* `model`: A saved MLM, without its file extension.
* `data`: A data directory, usually `data/test`
* `--step {time,quant}`: `time` (default) to use 40ms timesteps, `quant` to use 16th-note timesteps.
* `-wm weight_model`: The weight or prior model to use during testing.
* `-w FLOAT`: The weight to use for a constant weight model. Use `-w 1.0` to use the acoustic model only, thresholding at 0.5.
* `--save output_path`: Save results in the given directory.

**Important**: The results with 16th-note timesteps will be comparable to those in the paper. For the 40ms timestep, we use an additional post-processing step to remove short gaps. To do this, run evaluate.py with `--save output_path`, and then evaluate with: `python evaluate_load.py output_path data_path --gap`

## Contact

If you come across any issues or bugs, feel free to contact the authors, or report the issue directly in GitHub.




