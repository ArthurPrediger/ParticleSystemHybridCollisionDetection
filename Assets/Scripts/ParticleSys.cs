using Microsoft.Unity.VisualStudio.Editor;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static UnityEditor.PlayerSettings;

public class ParticleSys : MonoBehaviour
{
    [SerializeField]
    private ComputeShader PartSysUpdateCS;

    private Material partSysMat;

    private List<Vector3> particlesPos = new List<Vector3>();
    private List<Vector3> particlesVel = new List<Vector3>();

    private ComputeBuffer particlesPosCB;
    private ComputeBuffer particlesVelCB;

    RenderTexture depthTexture;

    public RawImage depthImage;

    // Start is called before the first frame update
    void Start()
    {
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        Bounds newBounds = new Bounds(Vector3.zero, Vector3.one * 999999f);
        meshRenderer.localBounds = newBounds;

        Vector3 starPos = new(0.5f, 0f, 0.5f);
        float offset = 1.0f;
        for(int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
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

        particlesPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3);
        particlesPosCB.SetData(particlesPos.ToArray());

        partSysMat = GetComponent<MeshRenderer>().material;
        partSysMat.SetBuffer("particlesPos", particlesPosCB);

        PartSysUpdateCS.SetBuffer(PartSysUpdateCS.FindKernel("PSUpdate"), "particlesPos", particlesPosCB);

        particlesVelCB = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3);
        particlesVelCB.SetData(particlesVel.ToArray());

        PartSysUpdateCS.SetBuffer(PartSysUpdateCS.FindKernel("PSUpdate"), "particlesVel", particlesVelCB);

        depthTexture = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.RFloat);
        depthTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        depthTexture.Create();

        PartSysUpdateCS.SetTexture(PartSysUpdateCS.FindKernel("PSUpdate"), "depthTexture", depthTexture);
    }

    // Update is called once per frame
    void Update()
    {
        DepthPrePass();

        depthImage.texture = depthTexture;

        PartSysUpdateCS.SetFloat(Shader.PropertyToID("deltaTime"), Time.deltaTime);

        PartSysUpdateCS.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        PartSysUpdateCS.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        PartSysUpdateCS.SetMatrix("inverseProjectionMat", Camera.main.projectionMatrix.inverse);
        PartSysUpdateCS.SetVector("cameraPos", Camera.main.transform.position);

        Vector2 screenRes = new(Screen.width, Screen.height);
        PartSysUpdateCS.SetVector("screenSize", screenRes);

        PartSysUpdateCS.Dispatch(PartSysUpdateCS.FindKernel("PSUpdate"), particlesPos.Count, 1, 1);
    }

    void OnDestroy()
    {
        particlesPosCB?.Release();
        particlesVelCB?.Release();
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
}
