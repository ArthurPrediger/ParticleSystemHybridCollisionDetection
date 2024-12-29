using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;

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

        MortonCodesBVHConstruction();
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
            particlesVelCB.SetData(particlesVel.ToArray());
            timerResetParticlesPos = 0;
        }
    }

    void OnDestroy()
    {
        particlesPosCB?.Release();
        particlesVelCB?.Release();
        particlesWithoutCollisionCB?.Release();
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

    private class BoundingBox
    {
        public Vector3 min = Vector3.positiveInfinity;
        public Vector3 max = Vector3.negativeInfinity;
        public Vector3 center = Vector3.zero;
        public Vector3 length = Vector3.zero;

        public void ScaleToInclude(Vector3 point)
        {
            min = Vector3.Min(min, point - Vector3.one * 0.01f);
            max = Vector3.Max(max, point + Vector3.one * 0.01f);
        }

        public void ScaleToInclude(List<Vector3> triangle)
        {
            foreach (Vector3 vertex in triangle)
            {
                ScaleToInclude(vertex);
            }
        }

        public static int TriangleMortonCode(List<Vector3> triangle, BoundingBox box)
        {
            if(triangle.Count != 3) return 0;

            const int gridSize = 1024;
            Vector3 gridUnitLength = box.length / gridSize;
            const float oneThird = 1f / 3f;

            Vector3 barycenter = (triangle[0] + triangle[1] + triangle[2]) * oneThird;
            Vector3 boxCoord = (barycenter - box.min);

            int xCoord = (int)Mathf.Floor(boxCoord.x / gridUnitLength.x);
            int yCoord = (int)Mathf.Floor(boxCoord.y / gridUnitLength.y);
            int zCoord = (int)Mathf.Floor(boxCoord.z / gridUnitLength.z);

            int mortonCode = 0;
            int mask = 0x00000001;
            int shiftAmount = 0;
            for (int j = 0; j < sizeof(int) * 8; j++)
            {
                mortonCode |= ((mask & xCoord) >> j) << shiftAmount++;
                mortonCode |= ((mask & yCoord) >> j) << shiftAmount++;
                mortonCode |= ((mask & zCoord) >> j) << shiftAmount++;
                mask <<= 1;
            }

            return mortonCode;
        }
    }

    class BVHTriangle : IComparable<BVHTriangle>
    {
        public Vector3[] vertices = new Vector3[3];
        public int mortonCode = 0;

        public int CompareTo(BVHTriangle other)
        {
            if(other == null) return 1;

            return mortonCode.CompareTo(other.mortonCode);
        }
    }

    void MortonCodesBVHConstruction()
    {
        BoundingBox box = new BoundingBox();
        List<BVHTriangle> triangles = new List<BVHTriangle>();

        // Find all GameObjects in the scene
        List<GameObject> allObjects = FindObjectsOfType<GameObject>().ToList();
        
        // Iterate through each GameObject and get its MeshFilter component
        foreach (GameObject obj in allObjects)
        {
            if (obj.TryGetComponent(out MeshFilter meshFilter))
            {
                Mesh mesh = meshFilter.sharedMesh;
                int vertexIndex = 3;
                foreach (int i in mesh.triangles)
                {
                    Vector3 vertex = obj.transform.TransformPoint(mesh.vertices[i]);
                    box.ScaleToInclude(vertex);

                    if (vertexIndex >= 3)
                    {
                        triangles.Add(new BVHTriangle());
                        vertexIndex = 0;
                    }
                    triangles.Last().vertices[vertexIndex++] = vertex;
                }
            }
        }

        box.center = (box.max + box.min) * 0.5f;
        box.length = (box.max - box.min);

        Debug.Log("Number of triangles: " +  triangles.Count);

        foreach(BVHTriangle tri in triangles)
        {
            tri.mortonCode = BoundingBox.TriangleMortonCode(tri.vertices.ToList(), box);
        }

        triangles.Sort();
    }
}
