mpicxx -g -Wall -o heated_plate_MPI heated_plate_MPI.cpp
g++ heated_plate_ParallerFor.cpp -ldl -o heated_plate_ParallerFor -L. -lPF -lpthread
g++ -fopenmp heated_plate_openmp.cpp -o heated_plate_openmp
mpiexec -n 4 ./heated_plate_MPI
./heated_plate_openmp
./heated_plate_ParallerFor