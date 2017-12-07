#!/bin/bash
#
# Fetch the datasets for this model.
#

OMNIGLOT_URL=https://raw.githubusercontent.com/brendenlake/omniglot/master/python

mkdir tmp
trap 'rm -r tmp' EXIT

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -d data/omniglot ]; then
    mkdir tmp/omniglot
    for name in images_background images_evaluation
    do
        echo "Fetching omniglot/$name ..."
        curl -s "$OMNIGLOT_URL/$name.zip" > "tmp/$name.zip" || exit 1
        echo "Extracting omniglot/$name ..."
        unzip -q "tmp/$name.zip" -d tmp || exit 1
        rm "tmp/$name.zip" || exit 1
        mv tmp/$name/* tmp/omniglot
    done
    mv tmp/omniglot data/omniglot
fi
