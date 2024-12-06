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
    private ComputeShader PSScreenSpaceCollisionDetectionCS;
    private int kernelIDScrSpaceColDetc;
    [SerializeField]
    private ComputeShader PSVolumeStructureCollisionDetectionCS;
    private int kernelIDVolStructColDetc;

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

        // Compute buffer IDs setting
        kernelIDReacUpdate = PSReactionUpdateCS.FindKernel("PSReactionUpdate");
        kernelIDScrSpaceColDetc = PSScreenSpaceCollisionDetectionCS.FindKernel("PSScreenSpaceCollisionDetection");
        kernelIDVolStructColDetc = PSVolumeStructureCollisionDetectionCS.FindKernel("PSVolumeStructureCollisionDetection");

        // Particles Positions gpu buffer setting
        particlesPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3);
        particlesPosCB.SetData(particlesPos.ToArray());

        instancedParticlesMat.SetBuffer("particlesPos", particlesPosCB);
        PSReactionUpdateCS.SetBuffer(kernelIDReacUpdate, "particlesPos", particlesPosCB);
        PSScreenSpaceCollisionDetectionCS.SetBuffer(kernelIDScrSpaceColDetc, "particlesPos", particlesPosCB);
        PSVolumeStructureCollisionDetectionCS.SetBuffer(kernelIDVolStructColDetc, "particlesPos", particlesPosCB);

        // Particles Velocities gpu buffer setting
        particlesVelCB = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3);
        particlesVelCB.SetData(particlesVel.ToArray());

        PSReactionUpdateCS.SetBuffer(kernelIDReacUpdate, "particlesVel", particlesVelCB);
        PSScreenSpaceCollisionDetectionCS.SetBuffer(kernelIDScrSpaceColDetc, "particlesVel", particlesVelCB);
        PSVolumeStructureCollisionDetectionCS.SetBuffer(kernelIDVolStructColDetc, "particlesVel", particlesVelCB);

        // Particles without screen space collision detection gpu buffer setting
        particlesWithoutCollisionCB = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3);

        PSScreenSpaceCollisionDetectionCS.SetBuffer(kernelIDScrSpaceColDetc, "particlesVel", particlesVelCB);
        PSVolumeStructureCollisionDetectionCS.SetBuffer(kernelIDVolStructColDetc, "particlesWithoutCollision", particlesWithoutCollisionCB);

        // Depth buffer for depth pre-pass setting
        depthTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.RFloat);
        depthTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        depthTexture.Create();

        PSScreenSpaceCollisionDetectionCS.SetTexture(kernelIDScrSpaceColDetc, "depthTexture", depthTexture);

        // Normal buffer for normal pre-pass setting
        normalTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.ARGBFloat);
        normalTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        normalTexture.Create();

        PSScreenSpaceCollisionDetectionCS.SetTexture(kernelIDScrSpaceColDetc, "normalTexture", normalTexture);
    }

    // Update is called once per frame
    void Update()
    {
        // Screen Space Particle Collision setting and dispatch
        DepthPrePass();
        NormalPrePass();
        textureImage.texture = depthTexture;

        PSScreenSpaceCollisionDetectionCS.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        PSScreenSpaceCollisionDetectionCS.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        PSScreenSpaceCollisionDetectionCS.SetMatrix("inverseProjectionMat", Camera.main.projectionMatrix.inverse);
        PSScreenSpaceCollisionDetectionCS.SetVector("cameraPos", Camera.main.transform.position);
        PSScreenSpaceCollisionDetectionCS.SetFloat("particleRadius", particleRadius);

        Vector2 screenRes = new(Screen.width, Screen.height);
        PSScreenSpaceCollisionDetectionCS.SetVector("screenSize", screenRes);

        PSScreenSpaceCollisionDetectionCS.Dispatch(kernelIDScrSpaceColDetc, 1/*particlesPos.Count*/, 1, 1);

        // Volumes Strcuture Particle Collision setting and dispatch
        PSVolumeStructureCollisionDetectionCS.SetFloat("particleRadius", particleRadius);

        //PSVolumeStructureCollisionDetectionCS.Dispatch(kernelIDVolStructColDetc, 1 , 1, 1);

        // Partcle System reaction update setting and dispatch
        PSReactionUpdateCS.SetFloat(Shader.PropertyToID("deltaTime"), Time.deltaTime);
        PSReactionUpdateCS.Dispatch(kernelIDReacUpdate, 1/*particlesPos.Count*/, 1, 1);

        // Particles mesh instancing rendering
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
