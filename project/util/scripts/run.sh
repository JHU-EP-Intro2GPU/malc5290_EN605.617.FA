#!/bin/bash
echo "Generating Rectangular Matrices floating point"
echo "Generating data for rect [128 X 256] X [256 X 512] (double/float)..."
python3 random-dense-matrix.py 100 128 256 ../matrix-gen/mat128_256
python3 random-dense-matrix.py 100 256 512 ../matrix-gen/mat256_512 

echo "Generating data for rect [512 X 1024] X [1024 X 2048] (double/float)..."
python3 random-dense-matrix.py 100 512 1024 ../matrix-gen/mat512_1024
python3 random-dense-matrix.py 100 1024 2048 ../matrix-gen/mat1024_2048 

echo "Generating data for rect [2048 X 4096] X [4096 X 8192] (double/float)..."
python3 random-dense-matrix.py 100 2048 4096 ../matrix-gen/mat2048_4096
python3 random-dense-matrix.py 100 4096 8192 ../matrix-gen/mat4096_8192 


echo "Running mult for [128 X 256] X [256 X 512] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -d | python3 matrix-result-writer.py

done

echo "Running mult for [512 X 1024] X [1024 X 2048] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -d | python3 matrix-result-writer.py

done

echo "Running mult for [2049 X 4096] X [4096 X 8192] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -d | python3 matrix-result-writer.py

done

echo "Cleaning up generated matrices"
rm ../matrix-gen/*

echo "Generating Rectangular Matrices integer"
echo "Generating data for rect [128 X 256] X [256 X 512] (int/short)..."
python3 random-dense-matrix.py 100 128 256 ../matrix-gen/mat128_256 --int-max 10
python3 random-dense-matrix.py 100 256 512 ../matrix-gen/mat256_512 --int-max 10

echo "Generating data for rect [512 X 1024] X [1024 X 2048] (int/short)..."
python3 random-dense-matrix.py 100 512 1024 ../matrix-gen/mat512_1024 --int-max 10
python3 random-dense-matrix.py 100 1024 2048 ../matrix-gen/mat1024_2048 --int-max 10

echo "Generating data for rect [2048 X 4096] X [4096 X 8192] (double/float)..."
python3 random-dense-matrix.py 100 2048 4096 ../matrix-gen/mat2048_4096
python3 random-dense-matrix.py 100 4096 8192 ../matrix-gen/mat4096_8192 


echo "Running mult for [128 X 256] X [256 X 512] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat128_256 -b ../matrix-gen/mat256_512 -s | python3 matrix-result-writer.py

done

echo "Running mult for [512 X 1024] X [1024 X 2048] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512_1024 -b ../matrix-gen/mat1024_2048 -s | python3 matrix-result-writer.py

done

echo "Running mult for [2049 X 4096] X [4096 X 8192] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048_4096 -b ../matrix-gen/mat4096_8192 -s | python3 matrix-result-writer.py

done

echo "Cleaning up generated matrices"
rm ../matrix-gen/*


echo "Generating Square Matrices floating point"
echo "Generating data for square [512 X 512] X [512 X 512] (double/float)..."
python3 random-dense-matrix.py 100 512 512 ../matrix-gen/mat512a
python3 random-dense-matrix.py 100 512 512 ../matrix-gen/mat512b

echo "Generating data for square [1024 X 1024] X [1024 X 1024] (double/float)..."
python3 random-dense-matrix.py 100 1024 1024 ../matrix-gen/mat1024a
python3 random-dense-matrix.py 100 1024 1024 ../matrix-gen/mat1024b 

echo "Generating data for square [2048 X 2048] X [2048 X 20248] (double/float)..."
python3 random-dense-matrix.py 100 2048 2048 ../matrix-gen/mat2048a
python3 random-dense-matrix.py 100 2048 2048 ../matrix-gen/mat2048b

echo "Generating data for square [4096 X 4096] X [4096 X 4096] (double/float)..."
python3 random-dense-matrix.py 100 4096 4096 ../matrix-gen/mat4096a
python3 random-dense-matrix.py 100 4096 4096 ../matrix-gen/mat4096b

echo "Generating data for square [8192 X 8192] X [8192 X 8192] (double/float)..."
python3 random-dense-matrix.py 100 8192 8192 ../matrix-gen/mat8192a
python3 random-dense-matrix.py 100 8192 8192 ../matrix-gen/mat8192b

echo "Running mult for [512 X 512] X [512 X 512] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py

done

echo "Running mult for [1024 X 1024] X [1024 X 1024] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py

done

echo "Running mult for [2048 X 2048] X [2048 X 2048] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
done

echo "Running mult for [4096 X 4096] X [4096 X 4096] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
done


echo "Running mult for [8192 X 8192] X [8192 X 8192] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
done


echo "Cleaning up generated matrices"
rm ../matrix-gen/*

echo "Generating Rectangular Matrices integer"
echo "Generating data for square [512 X 512] X [512 X 512] (int/short)..."
python3 random-dense-matrix.py 100 512 512 ../matrix-gen/mat512a --int-max 10
python3 random-dense-matrix.py 100 512 512 ../matrix-gen/mat512b --int-max 10


echo "Generating data for square [1024 X 1024] X [1024 X 1024] (int/short)..."
python3 random-dense-matrix.py 100 1024 1024 ../matrix-gen/mat1024a --int-max 10
python3 random-dense-matrix.py 100 1024 1024 ../matrix-gen/mat1024b  --int-max 10


echo "Generating data for square [2048 X 2048] X [2048 X 20248] (int/short)..."
python3 random-dense-matrix.py 100 2048 2048 ../matrix-gen/mat2048a --int-max 10
python3 random-dense-matrix.py 100 2048 2048 ../matrix-gen/mat2048b --int-max 10


echo "Generating data for square [4096 X 4096] X [4096 X 4096] (int/short)..."
python3 random-dense-matrix.py 100 4096 4096 ../matrix-gen/mat4096a --int-max 10
python3 random-dense-matrix.py 100 4096 4096 ../matrix-gen/mat4096b --int-max 10

echo "Generating data for square [8192 X 8192] X [8192 X 8192] (int/short)..."
python3 random-dense-matrix.py 100 8192 8192 ../matrix-gen/mat8192a --int-max 10
python3 random-dense-matrix.py 100 8192 8192 ../matrix-gen/mat8192b --int-max 10

echo "Running mult for [512 X 512] X [512 X 512] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -s | python3 matrix-result-writer.py

done

echo "Running mult for [1024 X 1024] X [1024 X 1024] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -s | python3 matrix-result-writer.py

done

echo "Running mult for [2048 X 2048] X [2048 X 2048] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -s | python3 matrix-result-writer.py
done

echo "Running mult for [4096 X 4096] X [4096 X 4096] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -s | python3 matrix-result-writer.py
done


echo "Running mult for [8192 X 8192] X [8192 X 8192] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -s | python3 matrix-result-writer.py
done

rm ../matrix-gen/*

mkdir dense-results
mv *.csv dense-results/.


echo "Generating Square Matrices floating point"
echo "Generating data for square [512 X 512] X [512 X 512] (double/float)..."
python3 random-dense-matrix.py 10 512 512 ../matrix-gen/mat512a
python3 random-dense-matrix.py 10 512 512 ../matrix-gen/mat512b

echo "Generating data for square [1024 X 1024] X [1024 X 1024] (double/float)..."
python3 random-dense-matrix.py 10 1024 1024 ../matrix-gen/mat1024a
python3 random-dense-matrix.py 10 1024 1024 ../matrix-gen/mat1024b 

echo "Generating data for square [2048 X 2048] X [2048 X 20248] (double/float)..."
python3 random-dense-matrix.py 10 2048 2048 ../matrix-gen/mat2048a
python3 random-dense-matrix.py 10 2048 2048 ../matrix-gen/mat2048b

echo "Generating data for square [4096 X 4096] X [4096 X 4096] (double/float)..."
python3 random-dense-matrix.py 10 4096 4096 ../matrix-gen/mat4096a
python3 random-dense-matrix.py 10 4096 4096 ../matrix-gen/mat4096b

echo "Generating data for square [8192 X 8192] X [8192 X 8192] (double/float)..."
python3 random-dense-matrix.py 10 8192 8192 ../matrix-gen/mat8192a
python3 random-dense-matrix.py 10 8192 8192 ../matrix-gen/mat8192b

echo "Running mult for [512 X 512] X [512 X 512] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -d | python3 matrix-result-writer.py

done

echo "Running mult for [1024 X 1024] X [1024 X 1024] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -d | python3 matrix-result-writer.py

done

echo "Running mult for [2048 X 2048] X [2048 X 2048] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -d | python3 matrix-result-writer.py
done

echo "Running mult for [4096 X 4096] X [4096 X 4096] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -d | python3 matrix-result-writer.py
done


echo "Running mult for [8192 X 8192] X [8192 X 8192] floating point..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -f | python3 matrix-result-writer.py
    ../../build/src/cublasmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -d | python3 matrix-result-writer.py
done


echo "Cleaning up generated matrices"
rm ../matrix-gen/*

echo "Generating Rectangular Matrices integer"
echo "Generating data for square [512 X 512] X [512 X 512] (int/short)..."
python3 random-dense-matrix.py 10 512 512 ../matrix-gen/mat512a --int-max 10
python3 random-dense-matrix.py 10 512 512 ../matrix-gen/mat512b --int-max 10


echo "Generating data for square [1024 X 1024] X [1024 X 1024] (int/short)..."
python3 random-dense-matrix.py 10 1024 1024 ../matrix-gen/mat1024a --int-max 10
python3 random-dense-matrix.py 10 1024 1024 ../matrix-gen/mat1024b  --int-max 10


echo "Generating data for square [2048 X 2048] X [2048 X 20248] (int/short)..."
python3 random-dense-matrix.py 10 2048 2048 ../matrix-gen/mat2048a --int-max 10
python3 random-dense-matrix.py 10 2048 2048 ../matrix-gen/mat2048b --int-max 10


echo "Generating data for square [4096 X 4096] X [4096 X 4096] (int/short)..."
python3 random-dense-matrix.py 10 4096 4096 ../matrix-gen/mat4096a --int-max 10
python3 random-dense-matrix.py 10 4096 4096 ../matrix-gen/mat4096b --int-max 10

echo "Generating data for square [8192 X 8192] X [8192 X 8192] (int/short)..."
python3 random-dense-matrix.py 10 8192 8192 ../matrix-gen/mat8192a --int-max 10
python3 random-dense-matrix.py 10 8192 8192 ../matrix-gen/mat8192b --int-max 10

echo "Running mult for [512 X 512] X [512 X 512] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat512a -b ../matrix-gen/mat512b -s | python3 matrix-result-writer.py

done

echo "Running mult for [1024 X 1024] X [1024 X 1024] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat1024a -b ../matrix-gen/mat1024b -s | python3 matrix-result-writer.py

done

echo "Running mult for [2048 X 2048] X [2048 X 2048] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat2048a -b ../matrix-gen/mat2048b -s | python3 matrix-result-writer.py
done

echo "Running mult for [4096 X 4096] X [4096 X 4096] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat4096a -b ../matrix-gen/mat4096b -s | python3 matrix-result-writer.py
done


echo "Running mult for [8192 X 8192] X [8192 X 8192] integer..."
for ((i = 0; i < 20; i++))
do
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -i | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -i | python3 matrix-result-writer.py
    ../../build/src/cutlassmult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -s | python3 matrix-result-writer.py
    ../../build/src/naivemult-driver -a ../matrix-gen/mat8192a -b ../matrix-gen/mat8192b -s | python3 matrix-result-writer.py
done

mkdir sparse-matrix
mv *.csv sparse-matrix/.

echo "final cleanup..."
rm ../matrix-gen/*
