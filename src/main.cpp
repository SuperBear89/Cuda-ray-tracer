#include <stdio.h>
#include "renderer.h"
#include <ctime>

extern cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{
	std::clock_t start;

	Renderer renderer;

	start = std::clock();
	renderer.loadSceneDesc();
	renderer.drawScene();
	printf("duration : %g s\n", ((double)std::clock() - (double)start) / (double)CLOCKS_PER_SEC);

	renderer.saveBmp("output.bmp");
	return 0;
}