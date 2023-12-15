#!/bin/bash
search_dir=./config/

for entry in "$search_dir"/*
do
  qsub launchers/launch.sh main.py $entry
done