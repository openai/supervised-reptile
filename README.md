# supervised-reptile

Reptile on Omniglot and (soon) miniImageNet.

# Getting the data

The [fetch_data.sh](fetch_data.sh) script creates a `data/` directory and downloads Omniglot and miniImageNet into it. The data is on the order of 5GB, so the download takes 10-20 minutes on a reasonably fast internet connection.

```
$ ./fetch_data.sh
Fetching omniglot/images_background ...
Extracting omniglot/images_background ...
Fetching omniglot/images_evaluation ...
Extracting omniglot/images_evaluation ...
Fetching miniImageNet train set ...
Fetching wnid: n01532829
Fetching wnid: n01558993
Fetching wnid: n01704323
Fetching wnid: n01749939
...
```

# Training on Omniglot

You can train a model on Omniglot with the `run_omniglot.py` script. If you don't specify any flags, the script uses a reasonable set of hyper-parameters. Here I've added some flags to make the script finish almost immediately at the expense of basically not learning anything:

```
$ python ./run_omniglot.py --meta-iters 10 --eval-samples 10
Training...
batch 0
batch 1
batch 2
batch 3
batch 4
batch 5
batch 6
batch 7
batch 8
batch 9
Evaluating...
Train accuracy: 0.7
Test accuracy: 0.6
```

The script also creates a `model_checkpoint` directory with the saved model and some summaries about accuracy during training. Once a checkpoint has been created, you can re-evaluate the trained model by passing the `--pretrained` flag to the script.
