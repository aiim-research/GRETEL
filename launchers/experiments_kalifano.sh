#!/bin/bash
type='sge' # sge | local
search_dir=$1 # the dir to search
trgt_scope='methods_comparison' # scope to be considered

for i in {1..30}
do
  for entry in "$search_dir"/*/*.json*
  do
    dcl_scope=$(python -c "import sys, json; from jsonc_parser.parser import JsoncParser; print(JsoncParser.parse_file('$entry')['experiment']['scope'])")

    if [ "$trgt_scope" == "$dcl_scope" ]; then
      echo "Founded $dcl_scope in $entry"
      if [ "$type" == "sge" ]; then
        sbatch launchers/launch.sh main.py $entry $i
      else
        python main.py $entry $i
      fi
    fi
  done
done