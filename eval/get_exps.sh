#!/usr/bin/env sh

seq 0 7 | xargs -P 8 -I % bash -c "gsutil -m cp -r gs://ganerator/ganerator-% ."

# Just download and put into correct folder.
mv --backup=t ganerator-*/*.tar.gz . && rmdir ganerator-*

