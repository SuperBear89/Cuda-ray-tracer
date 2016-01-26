#pragma once
#include "cuda_runtime.h"

struct Colorf
{
	float r;
	float g;
	float b;
};

struct Ray
{
	float3 position;
	float3 direction;
	Colorf accumulatedMaterial;
};

struct Plane
{
	float3 normal;
	float3 vertices[4];
};

struct Sphere
{
	float3 origin;
	float radius;
};

struct Circle
{
	float3 normal;
	float3 position;
	float radius;
};

enum materialType{ materialType_diffuse, materialType_reflective, materialType_emission };

struct Material
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	materialType type;
};

struct SceneObjects
{
	Plane* planes;
	size_t planeCount;
	Sphere* spheres;
	size_t sphereCount;
	Circle* circles;
	size_t circlesCount;
	Material* materials;
};

struct Collision
{
	float distance;
	float3 normal;
	unsigned int materialId;
};

struct Light
{
	Colorf color;
	float intensity;
};