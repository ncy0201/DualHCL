#!/bin/bash

PD=data/douban
PREFIX1=online
PREFIX2=offline
TRAINRATIO=0.8

python -u DualHCL.py \
        --s_edge ${PD}/${PREFIX1}.txt \
        --t_edge ${PD}/${PREFIX2}.txt \
        --gt_path ${PD}/node,split=${TRAINRATIO}.test.dict \
        --train_path ${PD}/node,split=${TRAINRATIO}.train.dict \
        --out_path ${PD}/embeddings \


# PD=data/twitter_youtube
# PREFIX1=twitter
# PREFIX2=youtube

# PD=data/twitter_foursquare
# PREFIX1=twitter
# PREFIX2=foursquare

# PD=data/dblp
# PREFIX1=dblp17
# PREFIX2=dblp19

# PD=data/wd
# PREFIX1=weibo
# PREFIX2=douban