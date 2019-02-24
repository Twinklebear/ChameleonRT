typedef unsigned int8 uint8;
typedef unsigned int uint32;

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

// Convert the RGBF32 tile to sRGB and write it to the RGBA8 framebuffer
export void tile_to_uint8(uniform float *uniform tile, uniform uint8 *uniform fb,
		const uniform uint32 fb_width, const uniform uint32 fb_height,
		const uniform uint32 tile_x, const uniform uint32 tile_y,
		const uniform uint32 tile_width, const uniform uint32 tile_height)
{
	foreach (i = 0 ... tile_width, j = 0 ... tile_height) {
		const uint32 tile_px = (j * tile_width + i) * 3;
		const uint32 fb_px = ((j + tile_y) * fb_width + i + tile_x) * 4;
		fb[fb_px] = linear_to_srgb(tile[tile_px]) * 255.f;
		fb[fb_px + 1] = linear_to_srgb(tile[tile_px + 1]) * 255.f;
		fb[fb_px + 2] = linear_to_srgb(tile[tile_px + 2]) * 255.f;
		fb[fb_px + 3] = 255;
	}
}

