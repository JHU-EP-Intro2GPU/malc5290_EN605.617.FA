make
printf "\nCreating Data\n"
python3 random-matrix.py 1024 1024 mat1024 -m 10
python3 random-matrix.py 512 512 mat512 -m 10
python3 random-matrix.py 256 256 mat256 -m 10
printf "\nPerforming 3 runs of each...\n"
printf "###########################################\n"
printf "Run 256 X 256 :\n"
printf "###########################################\n"
./assignment.exe mat256
./assignment.exe mat256
./assignment.exe mat256
printf "###########################################\n"
printf "Run 512 X 512:\n"
printf "###########################################\n"
./assignment.exe mat512
./assignment.exe mat512
./assignment.exe mat512
printf "###########################################\n"
printf "Run 1025 X 1024:\n"
printf "###########################################\n"
./assignment.exe mat1024
./assignment.exe mat1024
./assignment.exe mat1024
