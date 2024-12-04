using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ParticleSys : MonoBehaviour
{
    [SerializeField]
    private ComputeShader PSReactionUpdateCS;
    private int kernelIDReacUpdate;
    [SerializeField]
    private ComputeShader PScreenSpaceCollisionDetectionCS;
    private int kernelIDSSColDetc;

    [SerializeField]
    private Material instancedParticlesMat;
    [SerializeField]
    private Mesh particleMesh;
    private float particleRadius = 0.4f;

    private Material partSysMat;

    private List<Vector3> particlesPos = new();
    private List<Vector3> particlesVel = new();
    //private List<float> particlesAliveTime = new();
    //private List<float> particlesLifeSpan = new();

    private ComputeBuffer particlesPosCB;
    private ComputeBuffer particlesVelCB;
    private ComputeBuffer particlesWithoutCollisionCB;
    //private ComputeBuffer particlesInitPosCB;
    //private ComputeBuffer particlesAliveTimeCB;
    //private ComputeBuffer particlesLifeSpanCB;

    RenderTexture depthTexture;
    RenderTexture normalTexture;

    public RawImage textureImage;

    private float timerResetParticlesPos = 0f; 

    // Start is called before the first frame update
    void Start()
    {
        instancedParticlesMat.enableInstancing = true;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        meshRenderer.enabled = false;

        Vector3 starPos = new Vector3(1.5f, 0f, 1.5f) + transform.position;
        float offset = 1.0f;
        for(int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                particlesPos.Add((starPos - new Vector3(offset * i, 0.0f, offset * j)));
                particlesVel.Add(Vector3.zero);
            }
        }

        kernelIDReacUpdate = PSReactionUpdateCS.FindKernel("PSReactionUpdate");
        kernelIDSSColDetc = PScreenSpaceCollisionDetectionCS.FindKernel("PSScreenSpaceCollisionDetection");

        particlesPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3);
        particlesPosCB.SetData(particlesPos.ToArray());

        instancedParticlesMat.SetBuffer("particlesPos", particlesPosCB);
        PSReactionUpdateCS.SetBuffer(kernelIDReacUpdate, "particlesPos", particlesPosCB);
        PScreenSpaceCollisionDetectionCS.SetBuffer(kernelIDSSColDetc, "particlesPos", particlesPosCB);

        particlesVelCB = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3);
        particlesVelCB.SetData(particlesVel.ToArray());

        PSReactionUpdateCS.SetBuffer(kernelIDReacUpdate, "particlesVel", particlesVelCB);
        PScreenSpaceCollisionDetectionCS.SetBuffer(kernelIDSSColDetc, "particlesVel", particlesVelCB);

        depthTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.RFloat);
        depthTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        depthTexture.Create();

        PScreenSpaceCollisionDetectionCS.SetTexture(kernelIDSSColDetc, "depthTexture", depthTexture);

        normalTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.ARGBFloat);
        normalTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        normalTexture.Create();

        PScreenSpaceCollisionDetectionCS.SetTexture(kernelIDSSColDetc, "normalTexture", normalTexture);
    }

    // Update is called once per frame
    void Update()
    {
        DepthPrePass();
        NormalPrePass();
        textureImage.texture = depthTexture;

        PScreenSpaceCollisionDetectionCS.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        PScreenSpaceCollisionDetectionCS.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        PScreenSpaceCollisionDetectionCS.SetMatrix("inverseProjectionMat", Camera.main.projectionMatrix.inverse);
        PScreenSpaceCollisionDetectionCS.SetVector("cameraPos", Camera.main.transform.position);
        PScreenSpaceCollisionDetectionCS.SetFloat("particleRadius", particleRadius);

        Vector2 screenRes = new(Screen.width, Screen.height);
        PScreenSpaceCollisionDetectionCS.SetVector("screenSize", screenRes);

        PScreenSpaceCollisionDetectionCS.Dispatch(kernelIDSSColDetc, 1/*particlesPos.Count*/, 1, 1);

        PSReactionUpdateCS.SetFloat(Shader.PropertyToID("deltaTime"), Time.deltaTime);
        PSReactionUpdateCS.Dispatch(kernelIDReacUpdate, 1/*particlesPos.Count*/, 1, 1);

        RenderParams rp = new RenderParams(instancedParticlesMat);

        Matrix4x4[] instData = new Matrix4x4[particlesPos.Count];
        for (int i = 0; i < particlesPos.Count; ++i)
            instData[i] = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, particleRadius * 2f * Vector3.one);

        Graphics.RenderMeshInstanced(rp, particleMesh, 0, instData);

        timerResetParticlesPos += Time.deltaTime;
        if (timerResetParticlesPos > 4f)
        {
            particlesPosCB.SetData(particlesPos.ToArray());
            timerResetParticlesPos = 0;
        }
    }

    void OnDestroy()
    {
        particlesPosCB?.Release();
        particlesVelCB?.Release();
        depthTexture.Release();
        normalTexture.Release();
    }

    void DepthPrePass()
    {
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.targetTexture = depthTexture;

            mainCamera.RenderWithShader(Shader.Find("Custom/DepthPrePass"), null);

            mainCamera.targetTexture = null;
        }
    }

    void NormalPrePass()
    {
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.targetTexture = normalTexture;

            mainCamera.RenderWithShader(Shader.Find("Custom/NormalPrePass"), null);

            mainCamera.targetTexture = null;
        }
    }
}
