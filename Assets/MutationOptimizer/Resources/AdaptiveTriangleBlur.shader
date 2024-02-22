Shader "Custom/AdaptiveTriangleBlur"
{
	Properties
	{
		_MainTex("Texture", 2D) = "white" {}
	}
	SubShader
	{
		Cull Off
		ZWrite Off
		ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma target 5.0
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"
			#pragma multi_compile TRIANGLE_SOLID TRIANGLE_GRADIENT TRIANGLE_GAUSSIAN
			#pragma multi_compile SINGLE_COLOR SPHERICAL_HARMONICS_2 SPHERICAL_HARMONICS_3 SPHERICAL_HARMONICS_4
			#include "OptimizerPrimitives.hlsl"

			sampler2D _MainTex;
			float4 _MainTex_TexelSize;
			Texture2D<float4> _DepthIDBuffer;
			StructuredBuffer<PrimitiveData> _PrimitiveBuffer;
			float3 _CustomWorldSpaceCameraPos;
			float _CameraFovVRad;
			float4x4 _CameraMatrixVP;
			int _PrimitiveCount;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				return o;
			}

			// w0, w1, w2, and w3 are the four cubic B-spline basis functions
			float w0(float a)
			{
				return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
			}

			float w1(float a)
			{
				return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
			}

			float w2(float a)
			{
				return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
			}

			float w3(float a)
			{
				return (1.0 / 6.0) * (a * a * a);
			}

			// g0 and g1 are the two amplitude functions
			float g0(float a)
			{
				return w0(a) + w1(a);
			}

			float g1(float a)
			{
				return w2(a) + w3(a);
			}

			// h0 and h1 are the two offset functions
			float h0(float a)
			{
				return -1.0 + w1(a) / (w0(a) + w1(a));
			}

			float h1(float a)
			{
				return 1.0 + w3(a) / (w2(a) + w3(a));
			}

			float4 BicubicTex2DLod(sampler2D tex, float2 uv, float4 texelSize, float lod)
			{
				texelSize.zw /= pow(2.0, lod);
				texelSize.zw = floor(texelSize.zw);
				texelSize.xy = 1.0 / texelSize.zw;
				uv = uv * texelSize.zw + 0.5;
				float2 iuv = floor(uv);
				float2 fuv = frac(uv);

				float g0x = g0(fuv.x);
				float g1x = g1(fuv.x);
				float h0x = h0(fuv.x);
				float h1x = h1(fuv.x);
				float h0y = h0(fuv.y);
				float h1y = h1(fuv.y);
				float g0y = g0(fuv.y);
				float g1y = g1(fuv.y);

				float2 p0 = (float2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texelSize.xy;
				float2 p1 = (float2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texelSize.xy;
				float2 p2 = (float2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texelSize.xy;
				float2 p3 = (float2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texelSize.xy;

				return g0y * (g0x * tex2Dlod(tex, float4(p0, 0, lod)) +
					g1x * tex2Dlod(tex, float4(p1, 0, lod))) +
					g1y * (g0x * tex2Dlod(tex, float4(p2, 0, lod)) +
					g1x * tex2Dlod(tex, float4(p3, 0, lod)));
			}

			float4 frag(v2f i) : SV_Target
			{
				// Get triangle data
				uint2 pixelCoord = i.vertex.xy;
				float4 depthIDBufferFetch = _DepthIDBuffer[pixelCoord];
				uint primitiveID = asuint(depthIDBufferFetch.r);
				PrimitiveData primitiveData = _PrimitiveBuffer[primitiveID];
				float lodLevel = 0;

				if (primitiveID < _PrimitiveCount)
				{
					// Compute filter size based on shortest triangle edge length in screen space
					float4 temp = mul(_CameraMatrixVP, float4(primitiveData.positions[0], 1));
					float2 pixelPos0 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
					temp = mul(_CameraMatrixVP, float4(primitiveData.positions[1], 1));
					float2 pixelPos1 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
					temp = mul(_CameraMatrixVP, float4(primitiveData.positions[2], 1));
					float2 pixelPos2 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
					float minEdgeLengthPixels = min(length(pixelPos0 - pixelPos1), min(length(pixelPos1 - pixelPos2), length(pixelPos2 - pixelPos0)));
					float maxEdgeLengthPixels = max(length(pixelPos0 - pixelPos1), max(length(pixelPos1 - pixelPos2), length(pixelPos2 - pixelPos0)));
					float avgEdgeLengthPixels = (length(pixelPos0 - pixelPos1) + length(pixelPos1 - pixelPos2) + length(pixelPos2 - pixelPos0)) / 3.0;
					float trianglePixelArea = sqrt(Unsigned2DTriangleArea(pixelPos0, pixelPos1, pixelPos2));

					// Return blurred input
					lodLevel = log2(trianglePixelArea) * 0.666;
					lodLevel = int(lodLevel);
				}
				//lodLevel = 6;

				float4 blurredSample = BicubicTex2DLod(_MainTex, i.uv, _MainTex_TexelSize, lodLevel);
				return float4(blurredSample.rgb, 1);
			}
			ENDCG
		}
	}
}
