**Status:** Archive (code is provided as-is, no updates expected)

# supervised-reptile

[Reptile](https://arxiv.org/abs/1803.02999) training code for [Omniglot](https://github.com/brendenlake/omniglot) and [Mini-ImageNet](https://openreview.net/pdf?id=rJY0-Kcll).

Reptile is a meta-learning algorithm that finds a good initialization. It works by sampling a task, training on the sampled task, and then updating the initialization towards the new weights for the task.

# Getting the data

The [fetch_data.sh](fetch_data.sh) script creates a `data/` directory and downloads Omniglot and Mini-ImageNet into it. The data is on the order of 5GB, so the download takes 10-20 minutes on a reasonably fast internet connection.

```
$ ./fetch_data.sh
Fetching omniglot/images_background ...
Extracting omniglot/images_background ...
Fetching omniglot/images_evaluation ...
Extracting omniglot/images_evaluation ...
Fetching Mini-ImageNet train set ...
Fetching wnid: n01532829
Fetching wnid: n01558993
Fetching wnid: n01704323
Fetching wnid: n01749939
...
```

If you want to download Omniglot but not Mini-ImageNet, you can simply kill the script after it starts downloading Mini-ImageNet. The script automatically deletes partially-downloaded data when it is killed early.

# Reproducing training runs

You can train models with the `run_omniglot.py` and `run_miniimagenet.py` scripts. Hyper-parameters are specified as flags (see `--help` for a detailed list). Here are the commands used for the paper:

```shell
# transductive 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15t --transductive

# transductive 1-shot 5-way Mini-ImageNet.
python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15t --transductive

# 5-shot 5-way Mini-ImageNet.
python -u run_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m55

# 1-shot 5-way Mini-ImageNet.
python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15

# 5-shot 5-way Omniglot.
python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55

# 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15

# 1-shot 20-way Omniglot.
python -u run_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o120

# 5-shot 20-way Omniglot.
python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o520
```

Training creates checkpoints. Currently, you cannot resume training from a checkpoint, but you can re-run evaluation from a checkpoint by passing `--pretrained`. You can use TensorBoard on the checkpoint directories to see approximate learning curves during training and testing.

To evaluate with transduction, pass the `--transductive` flag. In this implementation, transductive evaluation is faster than non-transductive evaluation since it makes better use of batches.

# Comparing different inner-loop gradient combinations

Here are the commands for comparing different gradient combinations. The `--foml` flag indicates that only the final gradient should be used.

```shell
# Shared hyper-parameters for all experiments.
shared="--sgd --seed 0 --inner-batch 25 --learning-rate 0.003 --meta-step-final 0 --meta-iters 40000 --eval-batch 25 --eval-iters 5 --eval-interval 1"

python run_omniglot.py --inner-iters 1 --train-shots 5 --meta-step 0.25 --checkpoint g1_ckpt $shared | tee g1.txt

python run_omniglot.py --inner-iters 2 --train-shots 10 --meta-step 0.25 --checkpoint g1_g2_ckpt $shared | tee g1_g2.txt
python run_omniglot.py --inner-iters 2 --train-shots 10 --meta-step 0.125 --checkpoint half_g1_g2_ckpt $shared | tee half_g1_g2.txt
python run_omniglot.py --foml --inner-iters 2 --train-shots 10 --meta-step 0.25 --checkpoint g2_ckpt $shared | tee g2.txt

python run_omniglot.py --inner-iters 3 --train-shots 15 --meta-step 0.25 --checkpoint g1_g2_g3_ckpt $shared | tee g1_g2_g3.txt
python run_omniglot.py --inner-iters 3 --train-shots 15 --meta-step 0.08325 --checkpoint third_g1_g2_g3_ckpt $shared | tee third_g1_g2_g3.txt
python run_omniglot.py --foml --inner-iters 3 --train-shots 15 --meta-step 0.25 --checkpoint g3_ckpt $shared | tee g3.txt

python run_omniglot.py --foml --inner-iters 4 --train-shots 20 --meta-step 0.25 --checkpoint g4_ckpt $shared | tee g4.txt
python run_omniglot.py --inner-iters 4 --train-shots 20 --meta-step 0.25 --checkpoint g1_g2_g3_g4_ckpt $shared | tee g1_g2_g3_g4.txt
python run_omniglot.py --inner-iters 4 --train-shots 20 --meta-step 0.0625 --checkpoint fourth_g1_g2_g3_g4_ckpt $shared | tee fourth_g1_g2_g3_g4.txt

```
