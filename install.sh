#!/bin/bash

if ! [ -x "$(command -v unzip)" ]; then
    echo "'unzip' could not be found. Please install with \"sudo apt install unzip\". " >&2
    exit 1
fi

if ! [ -x "$(command -v curl)" ]; then
    echo "'curl' could not be found. Please install with \"sudo apt install curl\". " >&2
    exit 1
fi


while true; do
    read -p "Do you wish to download datasets and backbone checkpoints? [y/n]: " yn
    case $yn in
    [Yy]* )
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s" -o data.zip
        rm cookie
        rm -rf data
        unzip data.zip -d data
        rm data.zip; 
        break;;
    [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Do you wish to download pretrained model checkpoints? [y/n]: " yn
    case $yn in
    [Yy]* )
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1C5ag5X_gKR1IHW6fVAHdMggu7ilU1XbC" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1C5ag5X_gKR1IHW6fVAHdMggu7ilU1XbC" -o snapshots.zip
        rm cookie
        rm -rf snapshots
        unzip snapshots.zip -d snapshots
        rm snapshots.zip; 
        break;;
    [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

if ! [ -x "$(command -v conda)" ]; then
    echo "'conda' could not be found. Please install with Anaconda " >&2
    exit 1
fi

while true; do
    read -p "Do you wish to create conda environment? [y/n]: " yn
    case $yn in
    [Yy]* )
        source ~/.zshrc
        conda create -y -n uacanet python=3.8
        "${HOME}/anaconda3/envs/uacanet/bin/python" -m pip install -r requirements.txt
        break;;
    [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

clear