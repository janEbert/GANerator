#!/bin/bash

export MNT_DIR="ganerator-disk"
export DEVICE_ID="sdb"
export MNT_PATH="/mnt/disks/$MNT_DIR"

sudo mkdir -p $MNT_PATH
sudo mount -o discard,defaults /dev/$DEVICE_ID $MNT_PATH
sudo chmod a+w $MNT_PATH

cd ~
git clone https://github.com/janEbert/GANerator.git
cd GANerator

