g++ -g -Wall -fopenmp -o OMPGEMM OpenMPGEMM.cpp
g++ -g -Wall -fopenmp -o OMPGEMMS OpenMPGEMMS.cpp
g++ -g -Wall -fopenmp -o OMPGEMMD OpenMPGEMMD.cpp
g++ -g -Wall -o PthFor ParallelFor.cpp -lpthread
g++ -ggdb -Wall -shared -fpic -o libPF.so ParallelFor.cpp
g++ PFGEMM.cpp -ldl -o PFGEMM -L. -lPF -lpthread
