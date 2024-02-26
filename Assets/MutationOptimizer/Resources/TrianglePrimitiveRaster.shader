Shader "Custom/TrianglePrimitiveRaster"
{
	Properties
	{

	}
	SubShader
	{
		Tags { "RenderType" = "Opaque" }
		ZWrite On
		ZTest Less
		Cull Off
		//Conservative True

		Pass
		{
			CGPROGRAM
			#pragma target 5.0
			#pragma vertex Vertex
			#pragma fragment Fragment
			#include "UnityCG.cginc"
			#pragma multi_compile NORMAL_POSITIONS ALTERNATE_POSITIONS
			#pragma multi_compile TRIANGLE_SOLID TRIANGLE_GRADIENT TRIANGLE_GAUSSIAN
			#pragma multi_compile OPAQUE_RENDER SORTED_ALPHA_RENDER STOCHASTIC_ALPHA_RENDER
			#pragma multi_compile SINGLE_COLOR SPHERICAL_HARMONICS_2 SPHERICAL_HARMONICS_3 SPHERICAL_HARMONICS_4
			#include "OptimizerPrimitives.hlsl"

			// Uniforms
			int _CurrentFrame;
			int _PrimitiveCount;
			float _SimpleColorRender;
			float4x4 _CameraMatrixVP;
			float3 _CurrentCameraWorldPos;
			float _DebugTriangleView;


			// ========================== VERTEX SHADER ==========================
			StructuredBuffer<PrimitiveData> _PrimitiveBuffer;

			struct Varyings
			{
				float4 position : SV_POSITION;
				float3 barycentric : TEXCOORD0;
				float3 worldViewDir : TEXCOORD1;
				uint primitiveID : PRIMITIVEID;
			};

			Varyings Vertex(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
			{
				// Retrieve vertex attributes
				PrimitiveData primitiveData = _PrimitiveBuffer[instanceID];
				float3 worldPos = primitiveData.positions[vertexID];
				#ifdef ALTERNATE_POSITIONS
					worldPos += primitiveData.basePosition;
				#endif

				// Camera projection
				float4 clipPos = mul(_CameraMatrixVP, float4(worldPos, 1));

				// Get fragment barycentric coordinates
				float3 vertexBarycentric = float3(0, 0, 0);
				if (vertexID == 0)
					vertexBarycentric.x = 1.0;
				else if (vertexID == 1)
					vertexBarycentric.y = 1.0;
				else
					vertexBarycentric.z = 1.0;

				// Output
				Varyings output;
				output.position = clipPos;
				output.primitiveID = instanceID;
				output.barycentric = vertexBarycentric;
				output.worldViewDir = normalize(worldPos - _CurrentCameraWorldPos);
				return output;
			}



			// ========================== FRAGMENT SHADER ==========================
			float4 Fragment(Varyings input) : SV_Target
			{
				// For directly rendering the primitives
				if (_SimpleColorRender > 0.5)
				{
					// Fetch color
					PrimitiveData primitiveData = _PrimitiveBuffer[input.primitiveID];
					float4 color = FetchColorFromPrimitive(primitiveData, input.worldViewDir, input.barycentric.xy);
					if (_DebugTriangleView > 0.5)
						color.rgb = GetRandomColor(input.primitiveID);
					return float4(color.rgb, 1);
				}

				// For optimization rendering, Depth+ID
				float4 fragment = float4(asfloat(input.primitiveID), Float3ToFloat(input.worldViewDir), 0, Pack16FloatsTo32(input.barycentric.x, input.barycentric.y));
				return fragment;
			}
			ENDCG
		}
	}
	CustomEditor "Pcx.DiskMaterialInspector"
}
