#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "stdio.h"

#include "float.h"
#include "math.h"
#include "cutil_math.h"
#include "rayUtil.h"

#define INV_SQRT_THREE 0.577350269f
#define PI	3.14159265358979323846f


__device__ float3 createUniformDirectionInHemisphere(const float3 & normal, curandState* randState)
{
	float3 result;

	//Create a random coordinate in spherical space, then calculate the cartesian equivalent
	float z = curand_uniform(randState);
	float r = sqrt(1.0f - z * z);
	float phi = 2 * PI * curand_uniform(randState);
	float x = cosf(phi) * r;
	float y = sinf(phi) * r;

	// Find an axis that is not parallel to normal
	float3 majorAxis;
	if (abs(normal.x) < INV_SQRT_THREE) {
		majorAxis.x = 1.f;
	}
	else if (abs(normal.y) < INV_SQRT_THREE) {
		majorAxis.y = 1.f;
	}
	else {
		majorAxis.z = 1.f;
	}

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;

	// Transform from spherical coordinates to the cartesian coordinates space
	// we just defined above, then use the definition to transform to world space
	result = normalize(u * x + v * y + w * z);

	return result;
}

__global__ void initialiseRayKernel(Ray* const d_rays, const unsigned int width, const unsigned int height, const float3 focalPoint, const unsigned int rngSeed)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width) return;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= height) return;

	unsigned int i = y * width + x;

	curandState_t randState;
	curand_init(rngSeed + i, 0, 0, &randState);
	
	d_rays[i].position.x = x + curand_uniform(&randState);
	d_rays[i].position.y = y + curand_uniform(&randState);
	d_rays[i].position.z = 0;
	d_rays[i].direction = normalize(d_rays[i].position - focalPoint);
	d_rays[i].accumulatedMaterial.r = 1.0f;
	d_rays[i].accumulatedMaterial.g = 1.0f;
	d_rays[i].accumulatedMaterial.b = 1.0f;
}

__global__ void checkCollisionKernel(Light* const d_frameBuffer, const Ray* const d_rays, const size_t rayCount, const SceneObjects sceneObjects, unsigned int rngSeed)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= rayCount) return;

	curandState_t randState;
	curand_init(rngSeed + x, 0, 0, &randState);

	Ray startRay = d_rays[x];

	for (size_t i = 0; i < 8; i++)
	{
		Ray ray = startRay;

		bool recurse = true;
		while (recurse)
		{
			Collision nearestCollision;
			nearestCollision.distance = FLT_MAX;
			nearestCollision.materialId = 0;
			for (unsigned int i = 0; i < sceneObjects.planeCount; i++)
			{
				Plane plane = sceneObjects.planes[i];

				float ln = dot(ray.direction, plane.normal);
				if (ln >= 0.0f) continue; // plane & ray are parallel

				float rayDirectionScalar = dot(plane.vertices[0] - ray.position, plane.normal) / ln;
				if (rayDirectionScalar <= 0.0f) continue; // plane is behind ray

				float3 collisionPoint = ray.direction * rayDirectionScalar + ray.position;

				float3 perpendicular;
				perpendicular = cross(collisionPoint - plane.vertices[0], plane.vertices[1] - plane.vertices[0]);
				if (dot(plane.normal, perpendicular) < 0.f) continue;

				perpendicular = cross(collisionPoint - plane.vertices[1], plane.vertices[2] - plane.vertices[1]);
				if (dot(plane.normal, perpendicular) < 0.f) continue;

				perpendicular = cross(collisionPoint - plane.vertices[2], plane.vertices[3] - plane.vertices[2]);
				if (dot(plane.normal, perpendicular) < 0.f) continue;

				perpendicular = cross(collisionPoint - plane.vertices[3], plane.vertices[0] - plane.vertices[3]);
				if (dot(plane.normal, perpendicular) < 0.f) continue;

				if (rayDirectionScalar < nearestCollision.distance) // intersection
				{
					nearestCollision.distance = rayDirectionScalar;
					nearestCollision.materialId = i;
					nearestCollision.normal = plane.normal;
				}
			}

			for (unsigned int i = 0; i < sceneObjects.sphereCount; i++)
			{
				Sphere sphere = sceneObjects.spheres[i];

				float3 rayToSphere = ray.position - sphere.origin;
				float rayToSphere_scalar = dot(ray.direction, rayToSphere);

				float val = pow(rayToSphere_scalar, 2) - pow(length(rayToSphere), 2) + pow(sphere.radius, 2);
				if (val <= 0.f) continue; //Doesn't intersect

				rayToSphere_scalar = -rayToSphere_scalar - sqrt(val);
				if (rayToSphere_scalar <= 0.0f) continue;

				if (rayToSphere_scalar < nearestCollision.distance) // intersection
				{
					nearestCollision.distance = rayToSphere_scalar;
					nearestCollision.materialId = i + sceneObjects.planeCount;
					nearestCollision.normal = normalize(ray.position + nearestCollision.distance * ray.direction - sphere.origin);
				}
			}

			recurse = false;
			if (nearestCollision.distance < FLT_MAX)
			{
				Material material = sceneObjects.materials[nearestCollision.materialId];
				if (material.type == materialType_diffuse)
				{
					ray.position += nearestCollision.distance * ray.direction;
					ray.direction = createUniformDirectionInHemisphere(nearestCollision.normal, &randState);
					ray.accumulatedMaterial.r *= (float)material.r / 256.f;
					ray.accumulatedMaterial.g *= (float)material.g / 256.f;
					ray.accumulatedMaterial.b *= (float)material.b / 256.f;
					if (curand_uniform(&randState) < ray.accumulatedMaterial.r + ray.accumulatedMaterial.g + ray.accumulatedMaterial.b)
						recurse = true;
				}
				else if (material.type == materialType_emission)
				{
					d_frameBuffer[x].color.r += (float)material.r * ray.accumulatedMaterial.r / 256.f;
					d_frameBuffer[x].color.g += (float)material.g * ray.accumulatedMaterial.g / 256.f;
					d_frameBuffer[x].color.b += (float)material.b * ray.accumulatedMaterial.b / 256.f;
					d_frameBuffer[x].intensity += (ray.accumulatedMaterial.r + ray.accumulatedMaterial.g + ray.accumulatedMaterial.b) / 3.0f;
				}
				else if (material.type == materialType_reflective)
				{
					ray.position += nearestCollision.distance * ray.direction;
					ray.direction = reflect(ray.direction, nearestCollision.normal);
					if (ray.accumulatedMaterial.r + ray.accumulatedMaterial.g + ray.accumulatedMaterial.b > 0.25f)
						recurse = true;
				}
			}
			else
			{
				float3 sunDir; 
				sunDir.x = 0;
				sunDir.y = -1;
				sunDir.z = 0;
				if (dot(sunDir, ray.direction) > 0.95)
				{
					d_frameBuffer[x].color.r += (float)1024 * ray.accumulatedMaterial.r / 256.f;
					d_frameBuffer[x].color.g += (float)1024 * ray.accumulatedMaterial.g / 256.f;
					d_frameBuffer[x].color.b += (float)1024 * ray.accumulatedMaterial.b / 256.f;

				}
				else
				{
					d_frameBuffer[x].color.r += (float)201 * ray.accumulatedMaterial.r / 256.f;
					d_frameBuffer[x].color.g += (float)226 * ray.accumulatedMaterial.g / 256.f;
					d_frameBuffer[x].color.b += (float)255 * ray.accumulatedMaterial.b / 256.f;
				}
				d_frameBuffer[x].intensity += (ray.accumulatedMaterial.r + ray.accumulatedMaterial.g + ray.accumulatedMaterial.b) / 3.0f;
			}
		}
	}
}


__global__ void computeCollisionsKernel(const Collision* const d_collisions, Light* const d_frameBuffer, const size_t rayCount, const Material* const d_materials)
{
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= rayCount) return;

	if (d_collisions[pos].distance < FLT_MAX)
	{
		unsigned int materialId = d_collisions[pos].materialId;
		Material material = d_materials[materialId];
		d_frameBuffer[pos].color.r = material.r;
		d_frameBuffer[pos].color.g = material.g;
		d_frameBuffer[pos].color.b = material.b;
		d_frameBuffer[pos].intensity = 1;
	}
	else
	{
		d_frameBuffer[pos].color.r = 0;
		d_frameBuffer[pos].color.g = 0;
		d_frameBuffer[pos].color.b = 0;
		d_frameBuffer[pos].intensity = 0;
	}
}

__global__ void computeLightKernel(const Collision* const d_collisions, Light* const d_frameBuffer, const unsigned int width, const unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width) return;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= height) return;

	unsigned int pos = y * width + x;
	if (d_collisions[pos].distance < FLT_MAX)
	{
		d_frameBuffer[pos].color.r = 0;
		d_frameBuffer[pos].color.g = 0;
		d_frameBuffer[pos].color.b = 255;
		d_frameBuffer[pos].intensity = 1;
	}
	else
	{
		d_frameBuffer[pos].color.r = 0;
		d_frameBuffer[pos].color.g = 0;
		d_frameBuffer[pos].color.b = 0;
		d_frameBuffer[pos].intensity = 0;
	}
}

extern void cudaInitialiseRays(Ray* const d_rays, const unsigned int width, const unsigned int height, const float3 focalPoint, const unsigned int rngSeed)
{
	dim3 blockSize(32, 32);
	dim3 gridSize(ceil((float)width / (float)blockSize.x), ceil((float)height / (float)blockSize.y));
	initialiseRayKernel << <gridSize, blockSize >> >(d_rays, width, height, focalPoint, rngSeed);
}

extern void cudaCheckCollisions(Light* const d_frameBuffer, const Ray* const d_rays, const size_t rayCount, const SceneObjects sceneObjects, const unsigned int rngSeed)
{
	dim3 blockSize(128);
	dim3 gridSize(ceil((float)rayCount / (float)blockSize.x));
	checkCollisionKernel << <gridSize, blockSize >> >(d_frameBuffer, d_rays, rayCount, sceneObjects, rngSeed);
}

extern void cudaComputeCollisions(const Collision* const d_collisions, Light* const d_frameBuffer, const size_t rayCount, const Material* const d_materials)
{
	dim3 blockSize(256);
	dim3 gridSize(ceil((float)rayCount / (float)blockSize.x));
	computeCollisionsKernel << <gridSize, blockSize >> >(d_collisions, d_frameBuffer, rayCount, d_materials);
}

extern void cudaComputeLight(const Collision* const d_collisions, Light* const d_frameBuffer, const unsigned int frameBuffer_width, const unsigned int frameBuffer_height)
{
	dim3 blockSize(32, 32);
	dim3 gridSize(ceil((float)frameBuffer_width / (float)blockSize.x), ceil((float)frameBuffer_height / (float)blockSize.y));
	computeLightKernel << <gridSize, blockSize >> >(d_collisions, d_frameBuffer, frameBuffer_width, frameBuffer_height);
}
