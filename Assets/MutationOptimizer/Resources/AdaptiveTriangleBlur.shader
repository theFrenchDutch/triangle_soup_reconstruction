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
			int _OutputHeight;
			Texture2D<float4> _DepthIDBuffer;
			StructuredBuffer<PrimitiveData> _PrimitiveBuffer;
			float3 _CustomWorldSpaceCameraPos;
			float _CameraFovVRad;

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

			float4 frag(v2f i) : SV_Target
			{
				// Get triangle data
				uint2 pixelCoord = i.vertex.xy;
				float4 depthIDBufferFetch = _DepthIDBuffer[pixelCoord];
				uint primitiveID = asuint(depthIDBufferFetch.r);
				PrimitiveData primitiveData = _PrimitiveBuffer[primitiveID];

				// Compute world space pixel size at primitive pos for current view point
				float3 centerPos = (primitiveData.positions[0] + primitiveData.positions[1] + primitiveData.positions[2]) / 3.0;
				float d = length(centerPos - _WorldSpaceCameraPos);
				float screenSize = 2.0 * tan(_CameraFovVRad / 2.0) * d;
				float pixelSize = screenSize / (float)_OutputHeight;

				// Determine filter size in screen space
				float minSizeLength = min(length(primitiveData.positions[0] - primitiveData.positions[1]), min(length(primitiveData.positions[1] - primitiveData.positions[2]), length(primitiveData.positions[2] - primitiveData.positions[0])));
				float minSizeLengthPixels = minSizeLength / pixelSize;
				float lodLevel = log2(minSizeLengthPixels) * 0.5;

				// Return blurred input
				float4 blurredSample = tex2Dlod(_MainTex, float4(i.uv, 0, lodLevel));
				return blurredSample;
			}
			ENDCG
		}
	}
}
