Shader "Custom/InstancedParticleSpheres" 
{ 
	Properties 
	{ 
		_Color ("Color", Color) = (1, 1, 1, 1) 
	} 
	SubShader 
	{ 
		Tags 
		{ 
			"RenderType"="Opaque"
			"LightMode"="ForwardBase"
		}
		//Pass 
		//{ 
		//	CGPROGRAM 
		//	#pragma vertex vert 
		//	#pragma fragment frag 
		//	#pragma multi_compile_instancing 
		//	#include "UnityCG.cginc"

		//	struct appdata 
		//	{ 
		//		float4 vertex : POSITION; 
		//		UNITY_VERTEX_INPUT_INSTANCE_ID 
		//	};

		//	struct v2f 
		//	{ 
		//		float4 pos : SV_POSITION; 
		//	}; 
			
		//	float4 _Color; 
		//	StructuredBuffer<float3> particlesPos; 
			
		//	v2f vert (appdata v) 
		//	{ 
		//		UNITY_SETUP_INSTANCE_ID(v); 

		//		v2f o;  

		//		float3 particlePos = float3(0, 0, 0); 
		//		#ifdef UNITY_INSTANCING_ENABLED 
		//		particlePos = particlesPos[unity_InstanceID]; 
		//		#endif 
				
		//		o.pos = UnityWorldToClipPos(mul(unity_ObjectToWorld, v.vertex) + float4(particlePos, 0.0)); 
				
		//		return o; 
		//	}
			
		//	fixed4 frag (v2f i) : SV_Target 
		//	{ 
		//		return _Color; 
		//	} 
		//	ENDCG 
		//}

		Pass 
		{ 
			Name "ParticlesInstancingPass"

			CGPROGRAM 
			#pragma vertex vert 
			#pragma fragment frag 
			#pragma multi_compile_instancing 
			#include "UnityCG.cginc"
			#define UNITY_INDIRECT_DRAW_ARGS IndirectDrawIndexedArgs
	        #include "UnityIndirect.cginc"

			struct appdata 
			{ 
				float4 vertex : POSITION;
			};

			struct v2f 
			{ 
				float4 pos : SV_POSITION; 
			}; 
			
			float4 _Color; 
			StructuredBuffer<float3> particlesPos;
			uniform float particleRadius;
			
			v2f vert (appdata v, uint svInstanceID : SV_InstanceID)
			{ 
				InitIndirectDrawArgs(0);

				v2f o;  

				float3 particlePos = float3(0, 0, 0); 
				particlePos = particlesPos[GetIndirectInstanceID(svInstanceID)];
				
				v.vertex *= particleRadius;
				o.pos = UnityWorldToClipPos(mul(unity_ObjectToWorld, v.vertex) + float4(particlePos, 0.0));
				
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target 
			{ 
				return _Color; 
			}
			ENDCG 
		}
	}
}

