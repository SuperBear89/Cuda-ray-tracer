#pragma once
#include "cuda_runtime.h"
#include <math.h>
#include <vector>
#include "rayUtil.h"
#include "cutil_math.h"
#include "EasyBMP.h"
#include "curand.h"
#include "tinyxml2.h"

class Renderer
{
public:
	Renderer();
	~Renderer();
	void loadSceneDesc();
	void drawScene();
	void saveBmp(const char* const fileName);
private:
	bool isSetuped;

	unsigned int frameBuffer_width;
	unsigned int frameBuffer_height;

	float3 focalPoint;

	unsigned int quality;

	Light* h_frameBuffer;
	std::vector<Plane> h_planes;
	std::vector<Sphere> h_spheres;
	std::vector<Circle> h_circles;
	std::vector<char> h_planes_materialId;
	std::vector<Material> h_materials;

	Light* d_frameBuffer;
	Ray* d_rays;
	SceneObjects d_sceneObjects;
	//Collision* d_collisions; // 2d array for storing the distance for each tested collisions. If there's no collision, the cell's value is FLT_MAX
	
	//Temporary functions for testing
	void buildMeshes1();
	void buildMeshes2();
};

extern void cudaInitialiseRays(Ray* const d_rays, const unsigned int width, const unsigned int height, const float3 focalPoint, const unsigned int rngSeed);

extern void cudaCheckCollisions(Light* const d_frameBuffer, const Ray* const d_rays, const size_t rayCount, const SceneObjects sceneObjects, const unsigned int rngSeed);

extern void cudaComputeCollisions(const Collision* const d_collisions, Light* const d_frameBuffer, const size_t rayCount, const Material* const d_materials);

extern void cudaComputeLight(const Collision* const d_collisions, Light* const d_frameBuffer, const unsigned int frameBuffer_width, const unsigned int frameBuffer_height);