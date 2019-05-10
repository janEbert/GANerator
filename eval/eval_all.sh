#!/usr/bin/env sh

genpypath='../src/generate_pics.py'
fidscorepath='../../pytorch-fid/fid_score.py'
npzpath='./ffhq.npz'

mkdir pics

for file in $(ls); do
    intar=$(tar -tf $file)
    # this was very lazy, we can sort and get the name then which is
    # much more robust
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

