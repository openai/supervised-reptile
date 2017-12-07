# supervised-reptile

Reptile on Omniglot and (soon) miniImageNet.

# Running

First, download the data like so:

```
$ ./fetch_data.sh
Fetching omniglot/images_background ...
Extracting omniglot/images_background ...
Fetching omniglot/images_evaluation ...
Extracting omniglot/images_evaluation ...
```

This will create a `data/` directory containing Omniglot.

Now, you can train a model on Omniglot like so. Here I added some flags to make it run really fast (at the expense of basically not learning anything):

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

The above command also creates a `model_checkpoint` with the saved model and some summaries about accuracy during training.
