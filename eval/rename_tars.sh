#!/usr/bin/env sh

pygetnorm_path='/vol/gancomp/experiments/get_norms.py'

name_prefix='64'

files=$(ls -l | grep '\.tar\.gz$' | awk '{print $9}')

for file in $files; do
    param_file=$(tar -tf $file | grep '\.pt$')
    tar -xzf $file $param_file
    norms=$(python3 $pygetnorm_path $param_file)
    dnorm=$(echo $norms | awk '{print $2}')
    gnorm=$(echo $norms | awk '{print $3}')
    mv --backup=t $file "$name_prefix-$dnorm-$gnorm-$file"

    rm $param_file
done

rmdir GANerator_experiments

