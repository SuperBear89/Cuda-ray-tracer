#include "renderer.h"

#define DEGREE_TO_RAD 0.0174532925f

Renderer::Renderer()
{
	isSetuped = false;
}


Renderer::~Renderer()
{
	if (isSetuped){
		delete[] h_frameBuffer;

		cudaFree(d_frameBuffer);
		cudaFree(d_rays);
		cudaFree(d_sceneObjects.planes);
		cudaFree(d_sceneObjects.spheres);
		cudaFree(d_sceneObjects.materials);

		cudaDeviceReset();
	}
}

void Renderer::loadSceneDesc()
{
	tinyxml2::XMLDocument doc;
	doc.LoadFile("sceneDesc.xml");

	tinyxml2::XMLElement* imgDesc = doc.FirstChildElement("image");
	frameBuffer_width = atoi(imgDesc->Attribute("width"));
	frameBuffer_height = atoi(imgDesc->Attribute("height"));
	float fov = atof(imgDesc->Attribute("fov"));

	focalPoint.x = (float)frameBuffer_width / 2;
	focalPoint.y = (float)frameBuffer_height / 2;
	float halfAngle = fov / 2;
	focalPoint.z = -focalPoint.x / tan(halfAngle * DEGREE_TO_RAD);

	quality = atoi(imgDesc->Attribute("quality"));

	//Loading scene objects
	tinyxml2::XMLElement* sceneDesc = imgDesc->FirstChildElement("scene");
	tinyxml2::XMLElement* sceneObject = sceneDesc->FirstChildElement("object");
	do
	{
		std::string type = sceneObject->Attribute("type");
		if (type == "plane"){
			Plane plane;
			tinyxml2::XMLElement* pos = sceneObject->FirstChildElement("pos");
			for (int i = 0; i < 4; i++){
				plane.vertices[i].x = atof(pos->Attribute("x"));
				plane.vertices[i].y = atof(pos->Attribute("y"));
				plane.vertices[i].z = atof(pos->Attribute("z"));
				pos = pos->NextSiblingElement("pos");
			}

			plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
			h_planes.push_back(plane);
		}
		else if (type == "sphere"){
			Sphere sphere;
			sphere.radius = atof(sceneObject->Attribute("radius"));
			tinyxml2::XMLElement* pos = sceneObject->FirstChildElement("pos");
			sphere.origin.x = atof(pos->Attribute("x"));
			sphere.origin.y = atof(pos->Attribute("y"));
			sphere.origin.z = atof(pos->Attribute("z"));

			h_spheres.push_back(sphere);
		}

		Material material;
		tinyxml2::XMLElement* xmlMaterial = sceneObject->FirstChildElement("material");

		std::string materialType = xmlMaterial->Attribute("type");
		if (materialType == "diffuse")
			material.type = materialType_diffuse;
		else if (materialType == "reflective")
			material.type = materialType_reflective;
		else if (materialType == "emission")
			material.type = materialType_emission;

		material.r = (unsigned char)atoi(xmlMaterial->Attribute("r"));
		material.g = (unsigned char)atoi(xmlMaterial->Attribute("g"));
		material.b = (unsigned char)atoi(xmlMaterial->Attribute("b"));
		
		h_materials.push_back(material);

		sceneObject = sceneObject->NextSiblingElement("object");
	} while (sceneObject != NULL);

	//Memory allocations
	h_frameBuffer = new Light[frameBuffer_width * frameBuffer_height];

	cudaMalloc(&d_frameBuffer, frameBuffer_width * frameBuffer_height * sizeof(Light));
	cudaMalloc(&d_rays, frameBuffer_width * frameBuffer_height * sizeof(Ray));
	cudaMalloc(&d_sceneObjects.planes, h_planes.size() * sizeof(Plane));
	cudaMalloc(&d_sceneObjects.spheres, h_spheres.size() * sizeof(Sphere));
	cudaMalloc(&d_sceneObjects.materials, h_materials.size() * sizeof(Material));

	cudaMemcpy(d_sceneObjects.planes, &h_planes[0], h_planes.size() * sizeof(Plane), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sceneObjects.spheres, &h_spheres[0], h_spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sceneObjects.materials, &h_materials[0], h_materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	d_sceneObjects.planeCount = h_planes.size();
	d_sceneObjects.sphereCount = h_spheres.size();
	d_sceneObjects.circlesCount = h_circles.size();

	isSetuped = true;
}

unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void Renderer::drawScene()
{
	cudaDeviceSynchronize();

	unsigned int nbPass = 8 * pow(2, quality);

	for (size_t i = 0; i < nbPass / 8; i++)
	{
		cudaInitialiseRays(d_rays, frameBuffer_width, frameBuffer_height, focalPoint, WangHash(i));
		cudaCheckCollisions(d_frameBuffer, d_rays, frameBuffer_width * frameBuffer_height, d_sceneObjects, WangHash(i + rand()));
		cudaDeviceSynchronize();

		printf(" Pass %d / %d\n", (i + 1) * 8, nbPass);
	}
}

void Renderer::saveBmp(const char* const fileName)
{
	cudaDeviceSynchronize();
	cudaMemcpy(h_frameBuffer, d_frameBuffer, frameBuffer_width * frameBuffer_height * sizeof(Light), cudaMemcpyDeviceToHost);

	float maxIntensity_r = 0.f;
	float maxIntensity_g = 0.f;
	float maxIntensity_b = 0.f;

	float minIntensity_r = FLT_MAX;
	float minIntensity_g = FLT_MAX;
	float minIntensity_b = FLT_MAX;

	for (size_t i = 0; i < frameBuffer_width * frameBuffer_height; i++)
	{
		if (h_frameBuffer[i].color.r > maxIntensity_r) maxIntensity_r = h_frameBuffer[i].color.r;
		if (h_frameBuffer[i].color.g > maxIntensity_g) maxIntensity_g = h_frameBuffer[i].color.g;
		if (h_frameBuffer[i].color.b > maxIntensity_b) maxIntensity_b = h_frameBuffer[i].color.b;

		if (h_frameBuffer[i].color.r < minIntensity_r) minIntensity_r = h_frameBuffer[i].color.r;
		if (h_frameBuffer[i].color.g < minIntensity_g) minIntensity_g = h_frameBuffer[i].color.g;
		if (h_frameBuffer[i].color.b < minIntensity_b) minIntensity_b = h_frameBuffer[i].color.b;
	}

	BMP output;
	output.SetBitDepth(24);
	output.SetSize(frameBuffer_width, frameBuffer_height);
	for (size_t j = 0; j < frameBuffer_height; j++)
	{
		for (size_t i = 0; i < frameBuffer_width; i++)
		{
			RGBApixel outputPix;

			size_t pos = j * frameBuffer_width + i;
			Light inputLight = h_frameBuffer[pos];


			float logScale_r = log(1 + inputLight.color.r - minIntensity_r) / log(1 + maxIntensity_r - minIntensity_r);
			float logScale_g = log(1 + inputLight.color.g - minIntensity_g) / log(1 + maxIntensity_g - minIntensity_g);
			float logScale_b = log(1 + inputLight.color.b - minIntensity_b) / log(1 + maxIntensity_b - minIntensity_b);
			outputPix.Red = (ebmpBYTE)(255 * logScale_r);
			outputPix.Green = (ebmpBYTE)(255 * logScale_g);
			outputPix.Blue = (ebmpBYTE)(255 * logScale_b);

			//outputPix.Red = (ebmpBYTE)(255 * inputLight.color.r / maxIntensity_r);
			//outputPix.Green = (ebmpBYTE)(255 * inputLight.color.g / maxIntensity_g);
			//outputPix.Blue = (ebmpBYTE)(255 * inputLight.color.b / maxIntensity_b);

			output.SetPixel(i, j, outputPix);
		}
	}
	output.WriteToFile(fileName);
}

void Renderer::buildMeshes1()
{
	Plane plane;
	Material material;

	float top = (float)frameBuffer_height - (float)frameBuffer_width * 0.95;
	float bottom = (float)frameBuffer_height;
	float back = (float)frameBuffer_width;
	float front = -10.f;
	float left = -10.f;
	float right = (float)frameBuffer_width + 10.f;

	//Left plane
	plane.vertices[0].x = left;
	plane.vertices[0].y = top;
	plane.vertices[0].z = front;

	plane.vertices[1].x = left;
	plane.vertices[1].y = top;
	plane.vertices[1].z = back;

	plane.vertices[2].x = left;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = back;

	plane.vertices[3].x = left;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = front;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));

	h_planes.push_back(plane);

	material.r = 240;
	material.g = 32;
	material.b = 32;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Right plane
	plane.vertices[0].x = right;
	plane.vertices[0].y = top;
	plane.vertices[0].z = back;

	plane.vertices[1].x = right;
	plane.vertices[1].y = top;
	plane.vertices[1].z = front;

	plane.vertices[2].x = right;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = front;

	plane.vertices[3].x = right;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = back;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 32;
	material.g = 240;
	material.b = 32;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//back plane
	plane.vertices[0].x = left;
	plane.vertices[0].y = top;
	plane.vertices[0].z = back;

	plane.vertices[1].x = right;
	plane.vertices[1].y = top;
	plane.vertices[1].z = back;

	plane.vertices[2].x = right;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = back;

	plane.vertices[3].x = left;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = back;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 240;
	material.g = 240;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//front plane
	/*
	plane.vertices[0].x = right;
	plane.vertices[0].y = top;
	plane.vertices[0].z = front;

	plane.vertices[1].x = left;
	plane.vertices[1].y = top;
	plane.vertices[1].z = front;

	plane.vertices[2].x = left;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = front;

	plane.vertices[3].x = right;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = front;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 192;
	material.g = 192;
	material.b = 192;
	material.type = materialType_diffuse;
	h_materials.push_back(material);
	*/

	//top plane
	plane.vertices[0].x = left;
	plane.vertices[0].y = top;
	plane.vertices[0].z = front;

	plane.vertices[1].x = right;
	plane.vertices[1].y = top;
	plane.vertices[1].z = front;

	plane.vertices[2].x = right;
	plane.vertices[2].y = top;
	plane.vertices[2].z = back;

	plane.vertices[3].x = left;
	plane.vertices[3].y = top;
	plane.vertices[3].z = back;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 240;
	material.g = 240;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//bottom plane
	plane.vertices[0].x = left;
	plane.vertices[0].y = bottom;
	plane.vertices[0].z = back;

	plane.vertices[1].x = right;
	plane.vertices[1].y = bottom;
	plane.vertices[1].z = back;

	plane.vertices[2].x = right;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = front;

	plane.vertices[3].x = left;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = front;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 240;
	material.g = 240;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	/*
	//mirror front
	plane.vertices[0].x = left + 1;
	plane.vertices[0].y = bottom / 4;
	plane.vertices[0].z = back * 5 / 6;

	plane.vertices[1].x = right / 4;
	plane.vertices[1].y = bottom / 4;
	plane.vertices[1].z = back + 1;

	plane.vertices[2].x = right / 4 + 64;
	plane.vertices[2].y = bottom + 1;
	plane.vertices[2].z = back - 64;

	plane.vertices[3].x = left + 64;
	plane.vertices[3].y = bottom + 1;
	plane.vertices[3].z = back * 5 / 6 - 64;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 192;
	material.g = 192;
	material.b = 192;
	material.type = materialType_reflective;
	h_materials.push_back(material);

	//mirror back
	plane.vertices[3].x = left + 1;
	plane.vertices[3].y = bottom / 4;
	plane.vertices[3].z = back * 5 / 6;

	plane.vertices[2].x = right / 4;
	plane.vertices[2].y = bottom / 4;
	plane.vertices[2].z = back + 1;

	plane.vertices[1].x = right / 4 + 64;
	plane.vertices[1].y = bottom + 1;
	plane.vertices[1].z = back - 64;

	plane.vertices[0].x = left + 64;
	plane.vertices[0].y = bottom + 1;
	plane.vertices[0].z = back * 5 / 6 - 64;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 128;
	material.g = 128;
	material.b = 128;
	material.type = materialType_diffuse;
	h_materials.push_back(material);
	*/

	//light source
	/*
	float size = 3.f;
	plane.vertices[0].x = left + right / size;
	plane.vertices[0].y = top + 1;
	plane.vertices[0].z = front + back / size;

	plane.vertices[1].x = right - right / size;
	plane.vertices[1].y = top + 1;
	plane.vertices[1].z = front + back / size;

	plane.vertices[2].x = right - right / size;
	plane.vertices[2].y = top + 1;
	plane.vertices[2].z = back - back / size;

	plane.vertices[3].x = left + right / size;
	plane.vertices[3].y = top + 1;
	plane.vertices[3].z = back - back / size;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 232;
	material.g = 232;
	material.b = 232;
	material.type = materialType_emission;
	h_materials.push_back(material);
	*/

	//Sphere 1
	Sphere sphere;

	sphere.radius = (float)frameBuffer_width / 6;
	sphere.origin.x = left + right / 2;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + back / 2;

	h_spheres.push_back(sphere);

	material.r = 240;
	material.g = 240;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere 2
	/*
	sphere.radius = (float)frameBuffer_width / 10;
	sphere.origin.x = right - sphere.radius;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = back - sphere.radius;

	h_spheres.push_back(sphere);

	material.r = 32;
	material.g = 32;
	material.b = 32;
	material.type = materialType_emission;
	h_materials.push_back(material);
	*/

	//Sphere 3
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = left + right / 2;
	sphere.origin.y = top;
	sphere.origin.z = front + back / 2;

	h_spheres.push_back(sphere);

	material.r = 255;
	material.g = 255;
	material.b = 255;
	material.type = materialType_emission;
	h_materials.push_back(material);
}

void Renderer::buildMeshes2(){
	Plane plane;
	Sphere sphere;
	Material material;

	float top = (float)frameBuffer_height - (float)frameBuffer_width * 0.95;
	float bottom = (float)frameBuffer_height + 100;
	float back = (float)frameBuffer_width + 10.f;
	float front = -10.f;
	float left = -10.f;
	float right = (float)frameBuffer_width + 10.f;

	//bottom plane
	plane.vertices[0].x = -15000.f;
	plane.vertices[0].y = bottom;
	plane.vertices[0].z = 15000.f;

	plane.vertices[1].x = 15000.f;
	plane.vertices[1].y = bottom;
	plane.vertices[1].z = 15000.f;

	plane.vertices[2].x = 15000.f;
	plane.vertices[2].y = bottom;
	plane.vertices[2].z = -15000.f;

	plane.vertices[3].x = -15000.f;
	plane.vertices[3].y = bottom;
	plane.vertices[3].z = -15000.f;

	plane.normal = normalize(cross(plane.vertices[2] - plane.vertices[1], plane.vertices[1] - plane.vertices[0]));
	h_planes.push_back(plane);

	material.r = 128;
	material.g = 128;
	material.b = 128;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere front right
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = right - sphere.radius * 2.f;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + sphere.radius * 1.1f;

	h_spheres.push_back(sphere);

	material.r = 16;
	material.g = 240;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere front left
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = left + sphere.radius * 2.f;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + sphere.radius * 1.1f;

	h_spheres.push_back(sphere);

	material.r = 16;
	material.g = 16;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere back left
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = left + sphere.radius * 2.f;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = back - sphere.radius * 1.1f;

	h_spheres.push_back(sphere);

	material.r = 240;
	material.g = 16;
	material.b = 16;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere back right
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = right - sphere.radius * 2.f;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = back - sphere.radius * 1.1f;

	h_spheres.push_back(sphere);

	material.r = 240;
	material.g = 240;
	material.b = 16;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere center
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = left + right / 2;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + back / 2;

	h_spheres.push_back(sphere);

	material.r = 255;
	material.g = 255;
	material.b = 255;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere 1
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = left;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + back / 2;

	h_spheres.push_back(sphere);

	material.r = 240;
	material.g = 16;
	material.b = 240;
	material.type = materialType_diffuse;
	h_materials.push_back(material);

	//Sphere 2
	sphere.radius = (float)frameBuffer_width / 8;
	sphere.origin.x = right;
	sphere.origin.y = bottom - sphere.radius;
	sphere.origin.z = front + back / 2;

	h_spheres.push_back(sphere);

	material.r = 16;
	material.g = 240;
	material.b = 16;
	material.type = materialType_diffuse;
	h_materials.push_back(material);
}