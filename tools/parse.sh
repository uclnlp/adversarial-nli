#!/usr/bin/env bash

NBLOGFILES=`ls $@ | wc -w`

echo "Number of log files:" $NBLOGFILES

ls $@ | xargs grep "Best Valid" | awk '{ print $8 " " $0 }' | sort -nr
