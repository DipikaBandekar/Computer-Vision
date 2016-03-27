all: CImg.h Classifier.h NearestNeighbor.h svm.h pca.h haar.h a3.cpp
	g++ a3.cpp -o a3 -lX11 -lpthread -I. -Isiftpp -O3 siftpp/sift.cpp

clean:
	rm a3
