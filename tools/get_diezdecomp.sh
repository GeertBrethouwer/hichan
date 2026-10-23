#!/usr/bin/env bash
#
# Fetch diezDecomp (R. Diez Sanhueza, J. Peeters, P. Costa, Comput. Phys. Commun. 316 (2025) 109811;
# MIT licence) into dependencies/diezDecomp. Only the Fortran sources are used; they are compiled
# together with HiCHAN:  make ... DIEZDECOMP=1
#
set -e
cd "$(dirname "$0")/.."
REV=68a20e35e9e09d0db97bda5e431f08820a3c906d   # revision tested with HiCHAN
if [ ! -d dependencies/diezDecomp ]; then
  mkdir -p dependencies
  git clone https://github.com/Rafael10Diez/diezDecomp dependencies/diezDecomp
fi
git -C dependencies/diezDecomp fetch -q origin || true
git -C dependencies/diezDecomp checkout -q $REV
echo "diezDecomp $(git -C dependencies/diezDecomp rev-parse --short HEAD) in dependencies/diezDecomp"
