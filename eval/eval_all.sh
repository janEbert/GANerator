#!/usr/bin/env sh

genpypath='/mnt/disks/rwdisk/GANerator/src/generate_pics.py'
fidscorepath='/mnt/disks/rwdisk/pytorch-fid/fid_score.py'
npzpath='/mnt/disks/rwdisk/ffhq.npz'

mkdir pics

for file in $(ls); do
    intar=$(tar -tf $file)
    lateststeps=$(echo "$intar" | grep -o -P '\d*(?=steps\.tar)' | sort -n)
    latestintar=$(echo "$intar" | grep $lateststeps'_steps\.tar$')
    paramsintar=$(echo "$intar" | grep '\.pt$')
    # This creates the directory 'GANerator_experiments'
    tar -xzf $file $latestintar $paramsintar
    python $genpypath --models_file $latestintar --params_file $paramsintar --save_dir pics
    python $fidscorepath $npzpath pics -c 0 --name $file.txt
    gsutil cp $file.txt gs://ganerator/results/
    rm pics/*
    rm GANerator_experiments/*
done

rmdir pics

