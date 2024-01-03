#!/bin/bash
type='sge' # sge | local
search_dir=./config # the dir to search
trgt_scope='examples_configs' # scope to be considered

for entry in "$search_dir"/*.json*
do
  dcl_scope=$(python -c "import sys, json; from jsonc_parser.parser import JsoncParser; print(JsoncParser.parse_file('$entry')['experiment']['scope'])")

  if [ "$trgt_scope" == "$dcl_scope" ]; then
    echo "Founded $dcl_scope in $entry"
    if [ "$type" == "sge" ]; then
      qsub launchers/launch.sh main.py $entry
    else
      python main.py $entry
    fi
  fi
done