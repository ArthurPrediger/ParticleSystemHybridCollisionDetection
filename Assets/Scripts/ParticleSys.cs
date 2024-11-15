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

    public RawImage depthImage;

    // Start is called before the first frame update
    void Start()
    {
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        Bounds newBounds = new Bounds(Vector3.zero, Vector3.one * 999999f);
        meshRenderer.localBounds = newBounds;

        Vector3 starPos = new(1.5f, 0f, 1.5f);
        float offset = 1.0f;
        for(int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                particlesPos.Add((starPos - new Vector3(offset * i, 0.0f, offset * j)));
                particlesVel.Add(Vector3.zero);
            }
        }

        List<int> indices = new List<int>();
        for(int i = 0; i < particlesPos.Count; i++)
        {
            indices.Add(i);
        }

        Mesh partMesh = new Mesh();
        partMesh.name = "Particle Mesh";
        partMesh.SetVertices(particlesPos);
        partMesh.SetIndices(indices, MeshTopology.Points, 0);
        GetComponent<MeshFilter>().mesh = partMesh;

        for (int i = 0; i < particlesPos.Count; i++)
        {
            particlesPos[i] = transform.localToWorldMatrix.MultiplyPoint3x4(particlesPos[i]);
        }

        kernelIDReacUpdate = PSReactionUpdateCS.FindKernel("PSReactionUpdate");
        kernelIDSSColDetc = PScreenSpaceCollisionDetectionCS.FindKernel("PSScreenSpaceCollisionDetection");

        particlesPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3);
        particlesPosCB.SetData(particlesPos.ToArray());

        partSysMat = GetComponent<MeshRenderer>().material;
        partSysMat.SetBuffer("particlesPos", particlesPosCB);

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
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        meshRenderer.enabled = false;
        DepthPrePass();
        NormalPrePass();
        meshRenderer.enabled = true;

        PScreenSpaceCollisionDetectionCS.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        PScreenSpaceCollisionDetectionCS.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        PScreenSpaceCollisionDetectionCS.SetMatrix("inverseProjectionMat", Camera.main.projectionMatrix.inverse);
        PScreenSpaceCollisionDetectionCS.SetVector("cameraPos", Camera.main.transform.position);

        Vector2 screenRes = new(Screen.width, Screen.height);
        PScreenSpaceCollisionDetectionCS.SetVector("screenSize", screenRes);

        PScreenSpaceCollisionDetectionCS.Dispatch(kernelIDSSColDetc, particlesPos.Count, 1, 1);

        PSReactionUpdateCS.SetFloat(Shader.PropertyToID("deltaTime"), Time.deltaTime);
        PSReactionUpdateCS.Dispatch(kernelIDReacUpdate, particlesPos.Count, 1, 1);
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
