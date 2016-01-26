#Cuda ray tracer
This is a basic ray tracer I made using CUDA. To compile and run it you need a CUDA-capable GPU and the nvidia toolkit.
You can configure the rendering parameters in sceneDesc.xml. Here's an example of a blank scene with a green sphere in the middle :
```XML
<image width="1280" height="720" fov="75" quality="4">
	<scene>
		<!--floor-->
		<object type="plane">
			<pos x="0" y="720" z="1280"/>
			<pos x="1280" y="720" z="1280"/>
			<pos x="1280" y="720" z="0"/>
			<pos x="0" y="720" z="0"/>
			<material type="diffuse" r="64" g="64" b="64" />
		</object>
		<object type="sphere" radius="160">
			<pos x="660" y="560" z="640"/>
			<material type="diffuse" r="10" g="255" b="10" />
		</object>
	</scene>
</image>
```

CUDA installation : 
* windows : http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/
* linux : http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/

Libraries :
* tinyxml2 : https://github.com/leethomason/tinyxml2
* easybmp : http://easybmp.sourceforge.net/

