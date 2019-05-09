#!/usr/bin/env sh

genpypath='/mnt/disks/rwdisk/GANerator/src/generate_pics.py'
fidscorepath='/mnt/disks/rwdisk/pytorch-fid/fid_score.py'
npzpath='/mnt/disks/rwdisk/ffhq.npz'

for file in $(ls); do
    intar=$(tar -tf $file)
    latestintar=$(echo "$intar" | grep '8206_steps\.tar$')
    paramsintar=$(echo "$intar" | grep '\.pt$')
    # Both of these create the directory 'GANerator_experiments'
    tar -xzf $file $latestintar
    tar -xzf $file $paramsintar
    python $genpypath --models_file $latestintar --params_file $paramsintar --save_dir pics
    python $fidscorepath $npzpath pics -c 0 --name $file.txt
    rm pics/*
    rm GANerator_experiments/*
done

