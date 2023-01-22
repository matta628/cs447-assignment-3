CFLAGS = -g -Wall -Wextra -pedantic 

all: spheres

spheres: spheres.o
	g++ $(CFLAGS) -o spheres spheres.o -fopenmp -mavx2 
spheres.o: spheres.cpp
	g++ $(CFLAGS) -c -fopenmp -mavx2 spheres.cpp 

clean:
	rm spheres.o spheres

