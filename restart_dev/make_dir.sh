#!/bin/bash

basedir=$1

mkdir $basedir
mkdir $basedir/10

for i in `seq 0 100 20000`; do mkdir $basedir/$i; done;
