CXX=nvcc
CFLAGS=-c -g -G -arch sm_70
INCLUDE=-I ./include  
LIBPATH=-L ./lib   
LDFLAGS = -lshtns

#wenwen -lm -ldl -lfftw3 -lcusparse -lcusolver -lcublas -lcufft
#wangxin -lfftw3_mpi -lfftw3 -lshtns -I/beegfs/home/wangxin/include -L/beegfs/home/wangxin/lib
OBJECTS=  Initialization

# Data FftwToolkit BasicFunc Initialization\ 
# 		IntegrationToolkit MDESolver MemFree SCFTBaseABCstar GaussElimination
all:ab
./lib/lib%.so:./src/%.cpp
	$(CXX) $(CFLAGS) -o $@ $< $(INCLUDE) $(LDFLAGS)  

./lib/lib%.so:./src/%.cu
	$(CXX) $(CFLAGS) -o $@ $< $(INCLUDE) $(LDFLAGS) 

ab:$(addsuffix .so, $(addprefix ./lib/lib, $(OBJECTS))) ab.cpp
	$(CXX) -g $(INCLUDE) ab.cpp -o $@ $(LIBPATH) $(addprefix -l, $(OBJECTS)) $(LDFLAGS) 
clean: 
	-rm ./lib/*.so
	-rm ab
