#!/bin/bash

# To prevent accidentally overwriting or mounting another of your disks,
# you HAVE to enter the sizes of the disks to use here.
# DISK_SIZE is the disk size for the disk to create for each instance
# (5G by default).
# RO_DISK_SIZE is the disk size for the read-only disk which contains
# the dataset.
export DISK_SIZE="14G"
export RO_DISK_SIZE="20G"

export MNT_BASE_DIR="/mnt/disks"
export MNT_DIR="rwdisk"
export RO_MNT_DIR="ganerator-disk"
export MNT_PATH="$MNT_BASE_DIR/$MNT_DIR"
export RO_MNT_PATH="$MNT_BASE_DIR/$RO_MNT_DIR"
export DEVICE_ID=$(sudo lsblk | awk "{ if (\$4 == \"$DISK_SIZE\") {print \$1} }")
export RO_DEVICE_ID=$(sudo lsblk | awk "{ if (\$4 == \"$RO_DISK_SIZE\" && \$5 == 1) {print \$1} }" | tail -n 1)

# Check if exactly one device has been fonud.
if (( $(echo $DEVICE_ID | grep -c .) == 1 )); then
    # We only want to format the temporary disk!
    sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "/dev/$DEVICE_ID"
else
    # We are unsure and do not want to delete data.
    echo "Could not find a unique disk with the given DISK_SIZE!"
    echo "No disk will be formatted."
fi
sudo mkdir -p $MNT_PATH
sudo mkdir -p $RO_MNT_PATH
sudo mount -o discard,defaults /dev/$DEVICE_ID $MNT_PATH
sudo mount -o discard,defaults /dev/$RO_DEVICE_ID $RO_MNT_PATH
sudo chmod a+w $MNT_PATH
sudo chmod a+w $RO_MNT_PATH

