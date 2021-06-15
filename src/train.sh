#!/usr/bin/bash

usage_exit() {
    echo "Usage: $0 [-l dir/to/save/log] [-c path/to/model/ckpt]" 1>&2
    exit 1
}

LOG_PATH='out.log'
CKPT_PATH=''

while getopts l:c:h OPT
do
    case $OPT in
        l) LOG_PATH=$OPTARG;;
        c) CKPT_PATH=$OPTARG;;
        h) usage_exit;;
        \?) usage_exit;;
    esac
done

if CKPT_PATH=='' ; then
    nohup python3 train.py > $LOG_PATH &
else
    nohup python3 -u train.py -c $CKPT_PATH > $LOG_PATH &
fi