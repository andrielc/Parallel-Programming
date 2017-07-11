INCLUDE_PATH = /usr/local/cuda/include
LIB_PATH = /usr/local/cuda/lib
CUDA_CC = /usr/local/cuda/bin/nvcc
CUDA_CFLAGS = -I$(INCLUDE_PATH) -keep #-DGPU_BENCHMARK -DDEBUG -DBENCHMARK
CC = gcc
CFLAGS = -I$(INCLUDE_PATH) -L$(LIB_PATH) -lcudart -lstdc++ #-DDEBUG -DBENCHMARK

all: main md5.o clean test1 test2 test3

main: main.c main.h md5.o
	$(CC) main.c md5.o -o main $(CFLAGS)

md5.o: md5.cu main.h
	$(CUDA_CC) md5.cu -c -o md5.o $(CUDA_CFLAGS)

clean:
	rm *.o  *.fat* *.cudafe* *.i* *.cubin *.ptx *.module*

test1:
	./main wordlist.txt db_10.txt crack_pass_db_10.txt > output1.dat

test2:
	./main wordlist.txt db_50.txt crack_pass_db_50.txt > output2.dat

test3:
	./main wordlist.txt db_100.txt crack_pass_db_100.txt > output3.dat

.PHONY: clean test1 test2 test3
