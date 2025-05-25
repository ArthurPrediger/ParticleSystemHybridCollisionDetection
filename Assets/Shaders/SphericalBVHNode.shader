Shader "Unlit/SphericalBVHNode"
{
    // Properties
    // {
    //     _Color ("Color", Color) = (0,1,0,1)
    // }
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

            fixed4 _Color;

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
                float3 objectScale : TEXCOORD3;
            };

            float3 ObjectScale() 
            {
                return float3(
                    length(unity_ObjectToWorld._m00_m10_m20),
                    length(unity_ObjectToWorld._m01_m11_m21),
                    length(unity_ObjectToWorld._m02_m12_m22));
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldNormal = mul((float3x3)unity_ObjectToWorld, v.normal);
                o.cameraPos = _WorldSpaceCameraPos;
                o.objectScale = ObjectScale();
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 worldPos = i.worldPos;
                float3 worldNormal = normalize(i.worldNormal);
                float3 cameraDir = normalize(i.cameraPos - worldPos);

                // Get sphere's approximate radius
                float sphereRadius = length(unity_ObjectToWorld._m00_m10_m20);

                if (pow(1.2 - abs(dot(worldNormal, cameraDir)), 8) < 0.8 * (sphereRadius / 400.0))
                {
                    discard;
                }

                return _Color;
            }
            ENDCG
        }
    }
}
