#!/bin/bash

for i in {1..10}; do
  echo "Current run: $i"
  python fcs_simulate_3kernels.py --llamacpp_server
  cd /home/atif/FCS-GPU-benchmarks/unit-test3/build
  make clean
  if make -j8; then
    . /home/atif/FCS-GPU-benchmarks/unit-test3/build/x86_64-ubuntu2204-clang181-opt/setup.sh
    /home/atif/FCS-GPU-benchmarks/unit-test3/build/x86_64-ubuntu2204-clang181-opt/bin/runTFCSSimulation
  else
    echo "Test Failed"
  fi
  cd /home/atif/celloai-dev/tests
done
