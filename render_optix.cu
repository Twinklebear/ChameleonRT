#include <optix.h>
#include <optix_math.h>

rtDeclareVariable(uint2, pixel, rtLaunchIndex, );

rtBuffer<uchar4, 2> framebuffer;

RT_PROGRAM void perspective_camera() {
	framebuffer[pixel] = make_uchar4(255, 0, 0, 255);
}

