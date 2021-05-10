
image:image.c image.h
	gcc -g image.c -o image -lm

omp:image-omp.c image.h
	gcc -fopenmp -g image-omp.c -o image-omp -lm

pthr:image-pthr.c image.h
	gcc -g image-pthr.c -o image-pthr -lm -pthread

clean:
	rm -f image output.png image-omp output-omp.png image-pthr output-pthr.png
