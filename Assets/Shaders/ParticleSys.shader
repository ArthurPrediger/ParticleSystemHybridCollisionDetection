// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Unlit/ParticleSys"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma target 5.0
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            // make fog workw
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                uint vertexID : SV_VertexID;
                //float3 normal : NORMAL;
                //float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : POSITION;
                //float4 color : COLOR;
                //float3 normal : NORMAL;
                //float2 uv : TEXCOORD0;
                //float3 worldPos : POSITION0;
            };

            struct g2f
            {
                float4 pos : POSITION;
                //float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            StructuredBuffer<float3> particlesPos;

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;

                //o.pos = UnityObjectToClipPos(float4(particlesPos[v.vertexID], 1.0));
                //o.worldPos = mul(unity_ObjectToWorld, float4(particlesPos[v.vertexID], 1.0)).xyz;
                o.pos = mul(unity_ObjectToWorld, float4(particlesPos[v.vertexID], 1.0));

                return o;
            }

            [maxvertexcount(6)]
            void geom(point v2f input[1], inout TriangleStream<g2f> triStream)
            {
                float3 camPos = _WorldSpaceCameraPos;
                float billboardSize = 0.5f;

                // Get the center of the billboard (the original vertex position)
                float3 center = input[0].pos.xyz;

                // Calculate the view direction (from center to camera)
                float3 viewDir = normalize(center - camPos);

                // Get the right and up vectors for the billboard
                float3 up = float3(0, 1, 0); // Use world up as the initial up vector
                float3 right = normalize(cross(up, viewDir)) * billboardSize * 0.5;

                // Recompute the up vector after computing the right
                up = normalize(cross(viewDir, right)) * billboardSize * 0.5;

                // Define the four vertices of the billboard quad
                float3 v0 = center + right + up;   // Top right
                float3 v1 = center - right + up;   // Top left
                float3 v2 = center - right - up;   // Bottom left
                float3 v3 = center + right - up;   // Bottom right

                // Create two triangles for the quad (as a billboard)
        
                // First triangle
                g2f o;
                o.uv = float2(0, 0);
                o.pos = UnityWorldToClipPos(float4(v0, 1));
                triStream.Append(o);

                o.uv = float2(1, 0);
                o.pos = UnityWorldToClipPos(float4(v1, 1));
                triStream.Append(o);

                o.uv = float2(1, 1);
                o.pos = UnityWorldToClipPos(float4(v2, 1));
                triStream.Append(o);

                // Second triangle
                o.uv = float2(0, 0);
                o.pos = UnityWorldToClipPos(float4(v0, 1));
                triStream.Append(o);

                o.uv = float2(1, 1);
                o.pos = UnityWorldToClipPos(float4(v2, 1));
                triStream.Append(o);

                o.uv = float2(0, 1);
                o.pos = UnityWorldToClipPos(float4(v3, 1));
                triStream.Append(o);
            }

            fixed4 frag (g2f i) : SV_Target
            {
                fixed4 col = float4(0.0, 0.5, 0.0, 1.0);
                return col;
            }
            ENDCG
        }
    }
}
