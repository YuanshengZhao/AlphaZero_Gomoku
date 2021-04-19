export LIBRARY_PATH=$LIBRARY_PATH:./libtf/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./libtf/lib
g++ -I./libtf/include -L./libtf/lib -Wl,-rpath,./libtf/lib -O3 ag.cpp -ltensorflow -Wall -o ag_batch.exe
