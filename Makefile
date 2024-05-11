CC=nvcc

all: main

main: main.cu cpu.o single_thread.o multi_thread.o shared_memory.o
	$(CC) main.cu cpu.o single_thread.o multi_thread.o shared_memory.o -o main

cpu.o: cpu.c
	$(CC) -c cpu.c -o cpu.o

single_thread.o: single_thread.cu
	$(CC) -c single_thread.cu -o single_thread.o

multi_thread.o: multi_thread.cu
	$(CC) -c multi_thread.cu -o multi_thread.o

shared_memory.o: shared_memory.cu
	$(CC) -c shared_memory.cu -o shared_memory.o

clean:
	rm -f *.o main
