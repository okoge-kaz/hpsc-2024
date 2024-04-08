#!/bin/bash

THIS_DIR=$(cd $(dirname $0); pwd)

rsync -avz \
  -e ssh \
  --delete \
  tsubame:/gs/bs/tga-bayes-crest/fujii/hpsc-2024/ \
  $THIS_DIR/
