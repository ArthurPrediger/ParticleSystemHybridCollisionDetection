Shader "Unlit/SphericalBVHNode"
{
    Properties
    {
        _Color ("Color", Color) = (0,1,0,1)
    }
    SubShader
    {
        Tags { "Queue"="AlphaTest" "RenderType"="TransparentCutout" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 worldNormal : TEXCOORD1;
                float3 cameraPos : TEXCOORD2;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldNormal = mul((float3x3)unity_ObjectToWorld, v.normal);
                o.cameraPos = _WorldSpaceCameraPos;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 worldPos = i.worldPos;
                float3 worldNormal = normalize(i.worldNormal);
                float3 cameraDir = normalize(i.cameraPos - worldPos);

                if((1 - abs(dot(worldNormal, cameraDir))) < 0.75)
                {
                    discard;
                }

                fixed4 col = float4(0, 1, 0, 1);
                return col;
            }
            ENDCG
        }
    }
}
