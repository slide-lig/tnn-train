#!/usr/bin/env bash
filepath=$1
if [ "$(uname)" == "Darwin" ]; then
  md5 -r $filepath | awk '{print $1}';
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  md5sum $filepath | awk '{print $1}';
fi
