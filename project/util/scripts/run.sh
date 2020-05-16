#!/bin/bash
echo "Generating Matrices"
#echo "Generating data for [128 X 256] X [256 X 512]..."
#python3 ../matrix-gen/random-matrix.py 128 256 ../matrix-gen/mat128_256
#python3 ../matrix-gen/random-matrix.py 256 512 ../matrix-gen/mat256_512 

echo "Generating data for [512 X 1024] X [1024 X 2048]..."
python3 random-matrix.py 512 1024 ../matrix-gen/mat512_1024
python3 random-matrix.py 1024 2048 ../matrix-gen/mat1024_2048 

#echo "Running mult for [128 X 256] X [256 X 512]..."
#for ((i = 1; i < 20; i++))
#do
#    ../../build/src/cublasmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -f | python3 ../matrix-gen/matrix-result-writer.py
#done

echo "Running mult for [512 X 1024] X [1024 X 2048]..."
for ((i = 1; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -f | python3 matrix-result-writer.py
done
