#!/bin/bash
wget 'https://www.dropbox.com/s/ub0yyw16f8ttjs5/my.h5?dl=1' -O './model.h5'
python3 ./test.py $1 $2 ./model.h5
exit 0