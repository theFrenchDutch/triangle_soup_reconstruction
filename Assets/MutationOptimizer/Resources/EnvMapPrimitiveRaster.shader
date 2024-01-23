Shader "Custom/EnvMapPrimitiveRaster"
{
	Properties
	{
	}
	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"
			#define ENVMAP_TEXEL
			#define SINGLE_COLOR
			#include "OptimizerPrimitives.hlsl"

			StructuredBuffer<PrimitiveData> _EnvMapPrimitiveBuffer;
			float4x4 _CameraInvVP;
			float3 _CurrentCameraWorldPos;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float2 uv : TEXCOORD0;
				float3 worldViewDir : TEXCOORD1;
			};

			v2f vert(appdata v)
			{
				float4 temp = mul(_CameraInvVP, float4(v.vertex.xy, 0.5, 1));
				float3 worldPos = temp.xyz / temp.w;
				float3 worldViewDir = normalize(worldPos - _CurrentCameraWorldPos);

				v2f o;
				o.vertex = v.vertex;
				o.uv = v.uv;
				o.worldViewDir = worldViewDir;
				return o;
			}

			float4 frag(v2f i) : SV_Target
			{
				//return float4(1, 0, 0, 1);
				return float4(SampleEnvMapPrimitiveBuffer(_EnvMapPrimitiveBuffer, i.worldViewDir), 1);
			}
			ENDCG
		}
	}
}
