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

			float4 Cubic(float v)
			{
				float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
				float4 s = n * n * n;
				float x = s.x;
				float y = s.y - 4.0 * s.x;
				float z = s.z - 4.0 * s.y + 6.0 * s.x;
				float w = 6.0 - x - y - z;
				return float4(x, y, z, w) * (1.0 / 6.0);
			}

			float4 tex2DBicubicLod(float2 texCoords, float lod)
			{
				float2 texSize = _MainTex_TexelSize.zw / pow(2.0, lod);
				float2 invTexSize = 1.0 / texSize;
				texCoords = texCoords * texSize - 0.5;

				float2 fxy = frac(texCoords);
				texCoords -= fxy;

				float4 xcubic = Cubic(fxy.x);
				float4 ycubic = Cubic(fxy.y);

				float4 c = texCoords.xxyy + float2(-0.5, 1.5).xyxy;

				float4 s = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
				float4 offset = c + float4(xcubic.yw, ycubic.yw) / s;

				offset *= invTexSize.xxyy;

				float4 sample0 = tex2Dlod(_MainTex, float4(offset.xz, 0, lod));
				float4 sample1 = tex2Dlod(_MainTex, float4(offset.yz, 0, lod));
				float4 sample2 = tex2Dlod(_MainTex, float4(offset.xw, 0, lod));
				float4 sample3 = tex2Dlod(_MainTex, float4(offset.yw, 0, lod));

				float sx = s.x / (s.x + s.y);
				float sy = s.z / (s.z + s.w);

				return lerp(lerp(sample3, sample2, sx), lerp(sample1, sample0, sx), sy);
			}

			float4 frag(v2f i) : SV_Target
			{
				// Get triangle data
				uint2 pixelCoord = i.vertex.xy;
				float4 depthIDBufferFetch = _DepthIDBuffer[pixelCoord];
				uint primitiveID = asuint(depthIDBufferFetch.r);
				PrimitiveData primitiveData = _PrimitiveBuffer[primitiveID];

				// Compute filter size based on shortest triangle edge length in screen space
				float4 temp = mul(_CameraMatrixVP, float4(primitiveData.positions[0], 1));
				float2 pixelPos0 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
				temp = mul(_CameraMatrixVP, float4(primitiveData.positions[1], 1));
				float2 pixelPos1 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
				temp = mul(_CameraMatrixVP, float4(primitiveData.positions[2], 1));
				float2 pixelPos2 = (temp.xy / temp.w * 0.5 + 0.5) * _MainTex_TexelSize.zw;
				float minEdgeLengthPixels = min(length(pixelPos0 - pixelPos1), min(length(pixelPos1 - pixelPos2), length(pixelPos2 - pixelPos0)));

				// Return blurred input
				float lodLevel = log2(minEdgeLengthPixels) * 0.66;
				lodLevel = int(lodLevel);
				//return float4(minEdgeLengthPixels.xxx / 1024, 1);
				float4 blurredSample = tex2DBicubicLod(i.uv, lodLevel);
				return float4(blurredSample.rgb, 1);
			}
			ENDCG
		}
	}
}
