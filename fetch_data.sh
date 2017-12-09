#!/bin/bash
#
# Fetch the datasets for this model.
#

OMNIGLOT_URL=https://raw.githubusercontent.com/brendenlake/omniglot/master/python
IMAGENET_URL=http://www.image-net.org/download/synset?release=latest&src=stanford

mkdir tmp
trap 'rm -r tmp' EXIT

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -d data/omniglot ]; then
    mkdir tmp/omniglot
    for name in images_background images_evaluation; do
        echo "Fetching omniglot/$name ..."
        curl -s "$OMNIGLOT_URL/$name.zip" > "tmp/$name.zip" || exit 1
        echo "Extracting omniglot/$name ..."
        unzip -q "tmp/$name.zip" -d tmp || exit 1
        rm "tmp/$name.zip" || exit 1
        mv tmp/$name/* tmp/omniglot
    done
    mv tmp/omniglot data/omniglot
fi

if [ ! -d data/miniimagenet ]; then
    if [ -z "$IMAGENET_USER" ] || [ -z "$IMAGENET_ACCESSKEY" ]; then
        echo "Set IMAGENET_USER and IMAGENET_ACCESSKEY to fetch miniImageNet." >&2
        echo >&2
        echo "To get an access key, go to this page and create an account," >&2
        echo "then search the page for 'accesskey':" >&2
        echo "http://image-net.org/download-images" >&2
        exit 1
    fi
    mkdir "tmp/miniimagenet"
    authed_url="${IMAGENET_URL}&username=$IMAGENET_USER&accesskey=$IMAGENET_ACCESSKEY"
    for subset in train test val; do
        mkdir "tmp/miniimagenet/$subset"
        echo "Fetching miniImageNet $subset set ..."
        list_file="metadata/imagenet_$subset.txt"
        for wnid in $(cat "$list_file" | cut -f 1 -d / | sort -u); do
            mkdir "tmp/miniimagenet/$subset/$wnid"
            echo "Fetching wnid $wnid ..."
            curl -s "${authed_url}&wnid=$wnid" >tmp/archive.tar
            mkdir "tmp/$wnid"
            tar xf tmp/archive.tar -C "tmp/$wnid"
            for name in $(grep $wnid "$list_file"); do
                mv "tmp/$name" "tmp/miniimagenet/$subset/$wnid"
            done
            rm -r "tmp/$wnid"
        done
    done
    mv "tmp/miniimagenet" "data/miniimagenet"
fi
