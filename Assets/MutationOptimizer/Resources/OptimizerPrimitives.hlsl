#define DO_GAMMA_COLORS
#define DO_CLAMP_COLORS

// ======================= UTILS =======================
float4 LinearToGammaRGB(float4 color)
{
	return color;
	return float4(pow(color.rgb, 1.0 / 2.2), color.a);
}

float Remap(float s, float a1, float a2, float b1, float b2)
{
	return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

float2 Remap(float2 s, float2 a1, float2 a2, float2 b1, float2 b2)
{
	return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

float3 Remap(float3 s, float3 a1, float3 a2, float3 b1, float3 b2)
{
	return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

float erfinv(float x)
{
	float w, p;
	w = -log((1.0 - x) * (1.0 + x));
	if (w < 5.000000)
	{
		w = w - 2.500000;
		p = 2.81022636e-08;
		p = 3.43273939e-07 + p * w;
		p = -3.5233877e-06 + p * w;
		p = -4.39150654e-06 + p * w;
		p = 0.00021858087 + p * w;
		p = -0.00125372503 + p * w;
		p = -0.00417768164 + p * w;
		p = 0.246640727 + p * w;
		p = 1.50140941 + p * w;
	}
	else
	{
		w = sqrt(w) - 3.000000;
		p = -0.000200214257;
		p = 0.000100950558 + p * w;
		p = 0.00134934322 + p * w;
		p = -0.00367342844 + p * w;
		p = 0.00573950773 + p * w;
		p = -0.0076224613 + p * w;
		p = 0.00943887047 + p * w;
		p = 1.00167406 + p * w;
		p = 2.83297682 + p * w;
	}
	return p * x;
}

float SampleNormalDistribution(float u, float mu, float sigma)
{
	//return mu + sigma * (sqrt(-2.0 * log(u.x))* cos(2.0 * pi * u.y));
	return sigma * 1.414213f * erfinv(2.0 * u.x - 1.0) + mu; // TODO : u should not be -1 or 1 for erfinv
}

uint WangHash(uint seed)
{
	// Actually PCG Hash
	uint state = seed * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint RandXorshift(inout uint rngState)
{
	// Actually PCG Hash
	uint state = rngState;
	rngState = rngState * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float GetRandomFloat(inout uint rngState)
{
	// -1/+1 mutations
	uint rand = RandXorshift(rngState);
	float res = float(rand) * (1.0 / 4294967296.0);
	return res > 0.5 ? 1.0 : -1.0;

	// Gaussian mutations
	/*RandXorshift(rngState);
	float res = float(rngState) * (1.0 / 4294967296.0);
	return SampleNormalDistribution(res, 0.0, 1.0);*/

	// Uniform mutations
	/*RandXorshift(rngState);
	float res = float(rngState) * (1.0 / 4294967296.0);
	return res * 2.0 - 1.0;*/
}

float Unsigned3DTriangleArea(float3 A, float3 B, float3 C)
{
	return length(cross(B - A, C - A));
}

float3 HSVtoRGB(float3 hsv)
{
	hsv.x = fmod(100.0 + hsv.x, 1.0);                                       // Ensure [0,1[

	float   HueSlice = 6.0 * hsv.x;                                            // In [0,6[
	float   HueSliceInteger = floor(HueSlice);
	float   HueSliceInterpolant = HueSlice - HueSliceInteger;                   // In [0,1[ for each hue slice

	float3  TempRGB = float3(hsv.z * (1.0 - hsv.y),
		hsv.z * (1.0 - hsv.y * HueSliceInterpolant),
		hsv.z * (1.0 - hsv.y * (1.0 - HueSliceInterpolant)));

	// The idea here to avoid conditions is to notice that the conversion code can be rewritten:
	//    if      ( var_i == 0 ) { R = V         ; G = TempRGB.z ; B = TempRGB.x }
	//    else if ( var_i == 2 ) { R = TempRGB.x ; G = V         ; B = TempRGB.z }
	//    else if ( var_i == 4 ) { R = TempRGB.z ; G = TempRGB.x ; B = V     }
	// 
	//    else if ( var_i == 1 ) { R = TempRGB.y ; G = V         ; B = TempRGB.x }
	//    else if ( var_i == 3 ) { R = TempRGB.x ; G = TempRGB.y ; B = V     }
	//    else if ( var_i == 5 ) { R = V         ; G = TempRGB.x ; B = TempRGB.y }
	//
	// This shows several things:
	//  . A separation between even and odd slices
	//  . If slices (0,2,4) and (1,3,5) can be rewritten as basically being slices (0,1,2) then
	//      the operation simply amounts to performing a "rotate right" on the RGB components
	//  . The base value to rotate is either (V, B, R) for even slices or (G, V, R) for odd slices
	//
	float  IsOddSlice = fmod(HueSliceInteger, 2.0);                          // 0 if even (slices 0, 2, 4), 1 if odd (slices 1, 3, 5)
	float  ThreeSliceSelector = 0.5 * (HueSliceInteger - IsOddSlice);          // (0, 1, 2) corresponding to slices (0, 2, 4) and (1, 3, 5)

	float3 ScrollingRGBForEvenSlices = float3(hsv.z, TempRGB.zx);           // (V, Temp Blue, Temp Red) for even slices (0, 2, 4)
	float3 ScrollingRGBForOddSlices = float3(TempRGB.y, hsv.z, TempRGB.x);  // (Temp Green, V, Temp Red) for odd slices (1, 3, 5)
	float3 ScrollingRGB = lerp(ScrollingRGBForEvenSlices, ScrollingRGBForOddSlices, IsOddSlice);

	float  IsNotFirstSlice = saturate(ThreeSliceSelector);                   // 1 if NOT the first slice (true for slices 1 and 2)
	float  IsNotSecondSlice = saturate(ThreeSliceSelector - 1.0);              // 1 if NOT the first or second slice (true only for slice 2)

	return lerp(ScrollingRGB.xyz, lerp(ScrollingRGB.zxy, ScrollingRGB.yzx, IsNotSecondSlice), IsNotFirstSlice);    // Make the RGB rotate right depending on final slice index
}

float Sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

float3 Sigmoid(float3 x)
{
	return 1.0 / (1.0 + exp(-x));
}

float4 Sigmoid(float4 x)
{
	return 1.0 / (1.0 + exp(-x));
}

float Pack16FloatsTo32(float a, float b)
{
	uint a16 = f32tof16(a);
	uint b16 = f32tof16(b);
	uint abPacked = (a16 << 16) | b16;
	return asfloat(abPacked);
}

void Unpack16FloatsFrom32(float input, out float a, out float b)
{
	uint uintInput = asuint(input);
	a = f16tof32(uintInput >> 16);
	b = f16tof32(uintInput);
}

float2 Unpack16FloatsFrom32(float input)
{
	uint uintInput = asuint(input);
	float a = f16tof32(uintInput >> 16);
	float b = f16tof32(uintInput);
	return float2(a, b);
}

float Float3ToFloat(float3 dir)
{
	// Assume dir is normalized
	// Quantize and shift the bits
	uint x = uint((dir.x * 0.5 + 0.5) * 1023.0); // 10 bits
	uint y = uint((dir.y * 0.5 + 0.5) * 1023.0); // 10 bits
	uint z = uint((dir.z * 0.5 + 0.5) * 1023.0); // 10 bits

	uint result = (x << 20) | (y << 10) | z; // Combine
	return asfloat(result);
}

float3 FloatToFloat3(float packedDir)
{
	uint packedDirUint = asuint(packedDir);
	float x = (float)((packedDirUint >> 20) & 1023) / 1023.0;
	float y = (float)((packedDirUint >> 10) & 1023) / 1023.0;
	float z = (float)(packedDirUint & 1023) / 1023.0;
	return float3(x, y, z) * 2.0 - 1.0;
}

float2 DirectionToElevationAzimuth(float3 dir)
{
	float azimuth = atan2(dir.y, dir.x);
	float elevation = atan2(sqrt(dir.x * dir.x + dir.y * dir.y), dir.z);
	return float2(azimuth, elevation);
}

float3 ElevationAzimuthToDirection(float2 azimuthElevation)
{
	float cosAzimuth = cos(azimuthElevation.x);
	float sinAzimuth = sin(azimuthElevation.x);
	float cosElevation = cos(azimuthElevation.y);
	float sinElevation = sin(azimuthElevation.y);

	float3 reconstructedDir;
	reconstructedDir.x = cosAzimuth * cosElevation;
	reconstructedDir.y = sinAzimuth * cosElevation;
	reconstructedDir.z = sinElevation;
	return reconstructedDir;
}




// ========================= PRIMITIVE SIZE =========================
#define ALPHA_SIZE 1

#define COLOR_SIZE 3
#ifdef SPHERICAL_HARMONICS_2
#define COLOR_SIZE 12
#endif
#ifdef SPHERICAL_HARMONICS_3
#define COLOR_SIZE 27
#endif
#ifdef SPHERICAL_HARMONICS_4
#define COLOR_SIZE 48
#endif

#define PRIMITIVE_SIZE COLOR_SIZE + ALPHA_SIZE
#ifdef TRIANGLE_SOLID
#define PRIMITIVE_SIZE 9 + COLOR_SIZE + ALPHA_SIZE
#endif
#ifdef TRIANGLE_GRADIENT
#define PRIMITIVE_SIZE 9 + COLOR_SIZE * 3 + ALPHA_SIZE * 3
#endif
#ifdef TRIANGLE_GAUSSIAN
#define PRIMITIVE_SIZE 9 + COLOR_SIZE + ALPHA_SIZE
#endif




// ======================= SPHERICAL HARMONICS =======================
static float SH_C0 = 0.28209479177387814;
static float SH_C1 = 0.4886025119029199;
static float SH_C2[5] =
{
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
static float SH_C3[7] =
{
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

struct Color
{
	float alpha;
	float3 sh0;
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	float3 sh1, sh2, sh3;
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	float3 sh4, sh5, sh6, sh7, sh8;
#endif
#if defined(SPHERICAL_HARMONICS_4)
	float3 sh9, sh10, sh11, sh12, sh13, sh14, sh15;
#endif
};

float _LearningRateColor;
float _LearningRateSH;
float _LearningRateAlpha;

Color ZeroInitColor()
{
	Color color;

	color.alpha = 0;
	color.sh0 = float3(0, 0, 0);
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	color.sh1 = float3(0, 0, 0);
	color.sh2 = float3(0, 0, 0);
	color.sh3 = float3(0, 0, 0);
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	color.sh4 = float3(0, 0, 0);
	color.sh5 = float3(0, 0, 0);
	color.sh6 = float3(0, 0, 0);
	color.sh7 = float3(0, 0, 0);
	color.sh8 = float3(0, 0, 0);
#endif
#if defined(SPHERICAL_HARMONICS_4)
	color.sh9 = float3(0, 0, 0);
	color.sh10 = float3(0, 0, 0);
	color.sh11 = float3(0, 0, 0);
	color.sh12 = float3(0, 0, 0);
	color.sh13 = float3(0, 0, 0);
	color.sh14 = float3(0, 0, 0);
	color.sh15 = float3(0, 0, 0);
#endif

	return color;
}

Color GetFloatArrayAsColor(float asArray[PRIMITIVE_SIZE], inout int offset)
{
	Color color;

	color.alpha = asArray[offset++];
	color.sh0 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	color.sh1 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh2 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh3 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	color.sh4 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh5 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh6 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh7 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh8 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
#endif
#if defined(SPHERICAL_HARMONICS_4)
	color.sh9 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh10 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh11 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh12 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh13 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh14 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
	color.sh15 = float3(asArray[offset++], asArray[offset++], asArray[offset++]);
#endif

	return color;
}

void GetColorAsFloatArray(Color color, inout float asArray[PRIMITIVE_SIZE], inout int offset)
{
	asArray[offset++] = color.alpha;
	asArray[offset++] = color.sh0.r; asArray[offset++] = color.sh0.g; asArray[offset++] = color.sh0.b;
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	asArray[offset++] = color.sh1.r; asArray[offset++] = color.sh1.g; asArray[offset++] = color.sh1.b;
	asArray[offset++] = color.sh2.r; asArray[offset++] = color.sh2.g; asArray[offset++] = color.sh2.b;
	asArray[offset++] = color.sh3.r; asArray[offset++] = color.sh3.g; asArray[offset++] = color.sh3.b;
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	asArray[offset++] = color.sh4.r; asArray[offset++] = color.sh4.g; asArray[offset++] = color.sh4.b;
	asArray[offset++] = color.sh5.r; asArray[offset++] = color.sh5.g; asArray[offset++] = color.sh5.b;
	asArray[offset++] = color.sh6.r; asArray[offset++] = color.sh6.g; asArray[offset++] = color.sh6.b;
	asArray[offset++] = color.sh7.r; asArray[offset++] = color.sh7.g; asArray[offset++] = color.sh7.b;
	asArray[offset++] = color.sh8.r; asArray[offset++] = color.sh8.g; asArray[offset++] = color.sh8.b;
#endif
#if defined(SPHERICAL_HARMONICS_4)
	asArray[offset++] = color.sh9.r; asArray[offset++] = color.sh9.g; asArray[offset++] = color.sh9.b;
	asArray[offset++] = color.sh10.r; asArray[offset++] = color.sh10.g; asArray[offset++] = color.sh10.b;
	asArray[offset++] = color.sh11.r; asArray[offset++] = color.sh11.g; asArray[offset++] = color.sh11.b;
	asArray[offset++] = color.sh12.r; asArray[offset++] = color.sh12.g; asArray[offset++] = color.sh12.b;
	asArray[offset++] = color.sh13.r; asArray[offset++] = color.sh13.g; asArray[offset++] = color.sh13.b;
	asArray[offset++] = color.sh14.r; asArray[offset++] = color.sh14.g; asArray[offset++] = color.sh14.b;
	asArray[offset++] = color.sh15.r; asArray[offset++] = color.sh15.g; asArray[offset++] = color.sh15.b;
#endif
}

void GetColorLearningRatesFloatArray(inout float asArray[PRIMITIVE_SIZE], inout int offset)
{
	asArray[offset++] = _LearningRateAlpha;
	asArray[offset++] = _LearningRateColor; asArray[offset++] = _LearningRateColor; asArray[offset++] = _LearningRateColor;

#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	float shDivider = 2.0f;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	shDivider = 4.0f;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
#endif
#if defined(SPHERICAL_HARMONICS_4)
	shDivider = 8.0f;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
	asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider; asArray[offset++] = _LearningRateColor / shDivider;
#endif
}

float4 FetchRGBAColor(Color colorData, float3 dir = float3(0, 0, 1))
{
	float4 colorRGBA;

	// Alpha
	colorRGBA.a = colorData.alpha;

	// Simple color
#if defined(SINGLE_COLOR)
	colorRGBA.rgb = colorData.sh0.rgb;
#endif

	// Spherical harmonic color
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	colorRGBA.rgb = SH_C0 * colorData.sh0;
	float x = dir.x; float y = dir.y; float z = dir.z;
	colorRGBA.rgb = colorRGBA.rgb + SH_C1 * (-y * colorData.sh1 + z * colorData.sh2 - x * colorData.sh3);
#endif
#if defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	float xx = x * x; float yy = y * y; float zz = z * z; float xy = x * y; float xz = x * z; float yz = y * z;
	colorRGBA.rgb = colorRGBA.rgb +
		SH_C2[0] * xy * colorData.sh4 +
		SH_C2[1] * yz * colorData.sh5 +
		SH_C2[2] * (2. * zz - xx - yy) * colorData.sh6 +
		SH_C2[3] * xz * colorData.sh7 +
		SH_C2[4] * (xx - yy) * colorData.sh8;
#endif
#if defined(SPHERICAL_HARMONICS_4)
	colorRGBA.rgb = colorRGBA.rgb +
		SH_C3[0] * y * (3. * xx - yy) * colorData.sh9 +
		SH_C3[1] * xy * z * colorData.sh10 +
		SH_C3[2] * y * (4. * zz - xx - yy) * colorData.sh11 +
		SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * colorData.sh12 +
		SH_C3[4] * x * (4. * zz - xx - yy) * colorData.sh13 +
		SH_C3[5] * z * (xx - yy) * colorData.sh14 +
		SH_C3[6] * x * (xx - 3. * yy) * colorData.sh15;
#endif
#if defined(SPHERICAL_HARMONICS_2) || defined(SPHERICAL_HARMONICS_3) || defined(SPHERICAL_HARMONICS_4)
	colorRGBA.rgb = max(colorRGBA.rgb + 0.5, 0.0.rrr);
#endif

	// Gamma
#ifdef DO_GAMMA_COLORS
	colorRGBA.rgb = pow(colorRGBA.rgb, 2.2);
#endif

	return colorRGBA;
}

Color PostProcessColor(Color colorData)
{
	colorData.alpha = clamp(colorData.alpha, 0.0, 1.0);

#ifndef SPHERICAL_HARMONICS_2
	colorData.sh0 = clamp(colorData.sh0, 0.0, 1.0);
#endif

	return colorData;
}




// ======================= TRIANGLE SOUP SOLID =======================
#ifdef TRIANGLE_SOLID
struct PrimitiveData
{
	float3 positions[3];
	Color color;
};

float _LearningRatePosition;

PrimitiveData ZeroInitPrimitiveData()
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(0, 0, 0);
	primitiveData.positions[1] = float3(0, 0, 0);
	primitiveData.positions[2] = float3(0, 0, 0);
	primitiveData.color = ZeroInitColor();
	return primitiveData;
}

PrimitiveData GetFloatArrayAsPrimitive(float asArray[PRIMITIVE_SIZE])
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(asArray[0], asArray[1], asArray[2]);
	primitiveData.positions[1] = float3(asArray[3], asArray[4], asArray[5]);
	primitiveData.positions[2] = float3(asArray[6], asArray[7], asArray[8]);
	int offset = 9;
	primitiveData.color = GetFloatArrayAsColor(asArray, offset);
	return primitiveData;
}

void GetPrimitiveAsFloatArray(PrimitiveData primitiveData, out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = primitiveData.positions[0].x;
	asArray[1] = primitiveData.positions[0].y;
	asArray[2] = primitiveData.positions[0].z;

	asArray[3] = primitiveData.positions[1].x;
	asArray[4] = primitiveData.positions[1].y;
	asArray[5] = primitiveData.positions[1].z;

	asArray[6] = primitiveData.positions[2].x;
	asArray[7] = primitiveData.positions[2].y;
	asArray[8] = primitiveData.positions[2].z;

	int offset = 9;
	GetColorAsFloatArray(primitiveData.color, asArray, offset);
}

void GetLearningRatesFloatArray(out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = _LearningRatePosition;
	asArray[1] = _LearningRatePosition;
	asArray[2] = _LearningRatePosition;
	asArray[3] = _LearningRatePosition;
	asArray[4] = _LearningRatePosition;
	asArray[5] = _LearningRatePosition;
	asArray[6] = _LearningRatePosition;
	asArray[7] = _LearningRatePosition;
	asArray[8] = _LearningRatePosition;
	int offset = 9;
	GetColorLearningRatesFloatArray(asArray, offset);
}

float4 FetchColorFromPrimitive(PrimitiveData primitiveData, float3 viewDir = float3(0, 0, 1), float2 barycentricXY = float2(0, 0))
{
	float4 color = FetchRGBAColor(primitiveData.color, viewDir);
	return color;
}

float3 GetPrimitiveWorldDepthSortPosition(PrimitiveData primitiveData)
{
	return (primitiveData.positions[0] + primitiveData.positions[1] + primitiveData.positions[2]) / 3.0;
}

void PostProcessPrimitive(inout PrimitiveData primitiveData)
{
	primitiveData.color = PostProcessColor(primitiveData.color);
}

bool PrimitiveIsValid(PrimitiveData primitiveData, out float primitiveArea, float minWorldArea)
{
	primitiveArea = Unsigned3DTriangleArea(primitiveData.positions[0], primitiveData.positions[1], primitiveData.positions[2]);
	if (primitiveArea < minWorldArea)
		return false;

#ifndef OPAQUE_RENDER
	if (primitiveData.color.alpha < 0.001)
		return false;
#endif

	return true;
}
#endif




// ======================= TRIANGLE SOUP GRADIENT =======================
#ifdef TRIANGLE_GRADIENT
struct PrimitiveData
{
	float3 positions[3];
	Color colors[3];
};

float _LearningRatePosition;

PrimitiveData ZeroInitPrimitiveData()
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(0, 0, 0);
	primitiveData.positions[1] = float3(0, 0, 0);
	primitiveData.positions[2] = float3(0, 0, 0);
	primitiveData.colors[0] = ZeroInitColor();
	primitiveData.colors[1] = ZeroInitColor();
	primitiveData.colors[2] = ZeroInitColor();
	return primitiveData;
}

PrimitiveData GetFloatArrayAsPrimitive(float asArray[PRIMITIVE_SIZE])
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(asArray[0], asArray[1], asArray[2]);
	primitiveData.positions[1] = float3(asArray[3], asArray[4], asArray[5]);
	primitiveData.positions[2] = float3(asArray[6], asArray[7], asArray[8]);
	int offset = 9;
	primitiveData.colors[0] = GetFloatArrayAsColor(asArray, offset);
	primitiveData.colors[1] = GetFloatArrayAsColor(asArray, offset);
	primitiveData.colors[2] = GetFloatArrayAsColor(asArray, offset);
	return primitiveData;
}

void GetPrimitiveAsFloatArray(PrimitiveData primitiveData, out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = primitiveData.positions[0].x;
	asArray[1] = primitiveData.positions[0].y;
	asArray[2] = primitiveData.positions[0].z;

	asArray[3] = primitiveData.positions[1].x;
	asArray[4] = primitiveData.positions[1].y;
	asArray[5] = primitiveData.positions[1].z;

	asArray[6] = primitiveData.positions[2].x;
	asArray[7] = primitiveData.positions[2].y;
	asArray[8] = primitiveData.positions[2].z;

	int offset = 9;
	GetColorAsFloatArray(primitiveData.colors[0], asArray, offset);
	GetColorAsFloatArray(primitiveData.colors[1], asArray, offset);
	GetColorAsFloatArray(primitiveData.colors[2], asArray, offset);
}

void GetLearningRatesFloatArray(out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = _LearningRatePosition;
	asArray[1] = _LearningRatePosition;
	asArray[2] = _LearningRatePosition;
	asArray[3] = _LearningRatePosition;
	asArray[4] = _LearningRatePosition;
	asArray[5] = _LearningRatePosition;
	asArray[6] = _LearningRatePosition;
	asArray[7] = _LearningRatePosition;
	asArray[8] = _LearningRatePosition;

	int offset = 9;
	GetColorLearningRatesFloatArray(asArray, offset);
	GetColorLearningRatesFloatArray(asArray, offset);
	GetColorLearningRatesFloatArray(asArray, offset);
}

float4 FetchColorFromPrimitive(PrimitiveData primitiveData, float3 viewDir = float3(0, 0, 1), float2 barycentricXY = float2(0, 0))
{
	float3 barycentric = float3(barycentricXY.x, barycentricXY.y, 1.0 - (barycentricXY.x + barycentricXY.y));
	float4 color = FetchRGBAColor(primitiveData.colors[0], viewDir) * barycentric.x
		+ FetchRGBAColor(primitiveData.colors[1], viewDir) * barycentric.y
		+ FetchRGBAColor(primitiveData.colors[2], viewDir) * barycentric.z;
	return color;
}

float3 GetPrimitiveWorldDepthSortPosition(PrimitiveData primitiveData)
{
	return (primitiveData.positions[0] + primitiveData.positions[1] + primitiveData.positions[2]) / 3.0;
}

void PostProcessPrimitive(inout PrimitiveData primitiveData)
{
	for (int i = 0; i < 3; i++)
		primitiveData.colors[i] = PostProcessColor(primitiveData.colors[i]);
}

bool PrimitiveIsValid(PrimitiveData primitiveData, out float primitiveArea, float minWorldArea)
{
	primitiveArea = Unsigned3DTriangleArea(primitiveData.positions[0], primitiveData.positions[1], primitiveData.positions[2]);
	if (primitiveArea < minWorldArea)
		return false;

#ifndef OPAQUE_RENDER
	if (primitiveData.colors[0].alpha < 0.001 && primitiveData.colors[1].alpha < 0.001 && primitiveData.colors[2].alpha < 0.001)
		return false;
#endif

	return true;
}
#endif




// ======================= TRIANGLE SOUP GAUSSIAN =======================
#ifdef TRIANGLE_GAUSSIAN
struct PrimitiveData
{
	float3 positions[3];
	Color color;
};

float _LearningRatePosition;

PrimitiveData ZeroInitPrimitiveData()
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(0, 0, 0);
	primitiveData.positions[1] = float3(0, 0, 0);
	primitiveData.positions[2] = float3(0, 0, 0);
	primitiveData.color = ZeroInitColor();
	return primitiveData;
}

PrimitiveData GetFloatArrayAsPrimitive(float asArray[PRIMITIVE_SIZE])
{
	PrimitiveData primitiveData;
	primitiveData.positions[0] = float3(asArray[0], asArray[1], asArray[2]);
	primitiveData.positions[1] = float3(asArray[3], asArray[4], asArray[5]);
	primitiveData.positions[2] = float3(asArray[6], asArray[7], asArray[8]);
	int offset = 9;
	primitiveData.color = GetFloatArrayAsColor(asArray, offset);
	return primitiveData;
}

void GetPrimitiveAsFloatArray(PrimitiveData primitiveData, out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = primitiveData.positions[0].x;
	asArray[1] = primitiveData.positions[0].y;
	asArray[2] = primitiveData.positions[0].z;

	asArray[3] = primitiveData.positions[1].x;
	asArray[4] = primitiveData.positions[1].y;
	asArray[5] = primitiveData.positions[1].z;

	asArray[6] = primitiveData.positions[2].x;
	asArray[7] = primitiveData.positions[2].y;
	asArray[8] = primitiveData.positions[2].z;

	int offset = 9;
	GetColorAsFloatArray(primitiveData.color, asArray, offset);
}

void GetLearningRatesFloatArray(out float asArray[PRIMITIVE_SIZE])
{
	asArray[0] = _LearningRatePosition;
	asArray[1] = _LearningRatePosition;
	asArray[2] = _LearningRatePosition;
	asArray[3] = _LearningRatePosition;
	asArray[4] = _LearningRatePosition;
	asArray[5] = _LearningRatePosition;
	asArray[6] = _LearningRatePosition;
	asArray[7] = _LearningRatePosition;
	asArray[8] = _LearningRatePosition;
	int offset = 9;
	GetColorLearningRatesFloatArray(asArray, offset);
}

float4 FetchColorFromPrimitive(PrimitiveData primitiveData, float3 viewDir = float3(0, 0, 1), float2 barycentricXY = float2(0, 0))
{
	float3 barycentric = float3(barycentricXY.x, barycentricXY.y, 1.0 - (barycentricXY.x + barycentricXY.y));
	float4 colorRGBA = FetchRGBAColor(primitiveData.color, viewDir);

	float alpha = colorRGBA.a * (27 * barycentric.x * barycentric.y * barycentric.z);
	alpha = pow(alpha, 3.0);

	return float4(colorRGBA.rgb, alpha);
}

float3 GetPrimitiveWorldDepthSortPosition(PrimitiveData primitiveData)
{
	return (primitiveData.positions[0] + primitiveData.positions[1] + primitiveData.positions[2]) / 3.0;
}

void PostProcessPrimitive(inout PrimitiveData primitiveData)
{
	primitiveData.color = PostProcessColor(primitiveData.color);
}

bool PrimitiveIsValid(PrimitiveData primitiveData, out float primitiveArea, float minWorldArea)
{
	primitiveArea = Unsigned3DTriangleArea(primitiveData.positions[0], primitiveData.positions[1], primitiveData.positions[2]);
	if (primitiveArea < minWorldArea)
		return false;

#ifndef OPAQUE_RENDER
	if (primitiveData.color.alpha < 0.01)
		return false;
#endif

	return true;
}
#endif