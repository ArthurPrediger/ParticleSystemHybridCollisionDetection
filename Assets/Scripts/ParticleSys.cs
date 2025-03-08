using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Runtime.CompilerServices;
using UnityEditor.Experimental.GraphView;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;

public class ParticleSys : MonoBehaviour
{
    [SerializeField]
    private ComputeShader psReactionUpdateCs;
    private int kernelIdReactUpdate;
    [SerializeField]
    private ComputeShader psScreenSpaceCollisionDetectionCs;
    private int kernelIdScrSpaceColDetc;
    [SerializeField]
    private ComputeShader psVolumeStructureCollisionDetectionCs;
    private int kernelIdVolStructColDetc;
    [SerializeField]
    private ComputeShader fillBufferCs;
    private int kernelIdFillBuffer;

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

    private ComputeBuffer particlesPosCb;
    private ComputeBuffer particlesVelCb;
    private ComputeBuffer particlesInitPosCB;
    private ComputeBuffer particlesAliveTimeCB;
    private ComputeBuffer particlesWithoutDepthCollisionCb;
    private ComputeBuffer counterBuffer;
    private ComputeBuffer bvhCb;
    private ComputeBuffer bvhStackCb; // Stack to hold nodes to be visited
    private ComputeBuffer bvhStackIndicesCb; // Per-thread index for stack handling
    private ComputeBuffer bvhTrianglesCb;
    //private ComputeBuffer particlesLifeSpanCB;

    RenderTexture depthTexture;
    RenderTexture normalTexture;

    public RawImage textureImage;

    private List<BvhTriangle> triangles = new();

    private List<BvhSphereNode> bvh = new();
    private const int numLevelsBVHMorton = 6;
    private const int maxLevelBvh = 16;
    private const int maxTrisPerBvhNode = 32;
    private int numLastLevelBvh = 0;
    private const int maxSahSamples = 64;

    [SerializeField]
    private GameObject sphericalNodePrefab;
    private List<GameObject> sphericalBvhNodes = new();
    private int bvhNodeLevelToRender = -1;
    private bool isRenderingLeaves = false;

    private List<BvhAabbNode> octree = new();
    private const int numLevelsOctreeMorton = 6;
    private const int maxLevelOctree = 16;
    private const int maxTrisPerOctreeNode = 32;
    private int numLastLevelOctree = 0;

    private const int threadGroupSize = 32;
    private const int bvhStackSizePerThread = 128;

    private bool isScreenSpaceCollisionActive = false;
    private bool isVolumeStructureCollisionActive = true;

    GraphicsBuffer commandBuf;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    const int commandCount = 1;

    // Start is called before the first frame update
    void Start()
    {
        instancedParticlesMat.enableInstancing = true;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        GetComponent<MeshRenderer>().enabled = false;
        GetComponent<MeshFilter>().mesh = null;

        int xzDimension = 64;
        float xzStart = (float)(xzDimension - 1) / 2f;
        Vector3 starPos = new Vector3(xzStart, 0f, xzStart) + transform.position;
        float offset = 1.0f;
        for (int i = 0; i < xzDimension; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < xzDimension; k++)
                {
                    particlesPos.Add((starPos - new Vector3(offset * i, -(offset * j * 4), offset * k)));
                    particlesVel.Add(Vector3.zero);
                }
            }
        }

        // Compute buffer IDs setting
        kernelIdReactUpdate = psReactionUpdateCs.FindKernel("PSReactionUpdate");
        kernelIdScrSpaceColDetc = psScreenSpaceCollisionDetectionCs.FindKernel("PSScreenSpaceCollisionDetection");
        kernelIdVolStructColDetc = psVolumeStructureCollisionDetectionCs.FindKernel("PSVolumeStructureCollisionDetection");
        kernelIdFillBuffer = fillBufferCs.FindKernel("FillBuffer");

        // Particles Positions gpu buffer setting
        particlesPosCb = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesPosCb.SetData(particlesPos);

        instancedParticlesMat.SetBuffer("particlesPos", particlesPosCb);
        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesPos", particlesPosCb);
        psScreenSpaceCollisionDetectionCs.SetBuffer(kernelIdScrSpaceColDetc, "particlesPos", particlesPosCb);
        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "particlesPos", particlesPosCb);

        // Particles Initial Positions gpu buffer setting
        particlesInitPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesInitPosCB.SetData(particlesPos);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesInitPos", particlesInitPosCB);

        // Particles Initial Positions gpu buffer setting
        particlesAliveTimeCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesAliveTimeCB.SetData(particlesVel);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesAliveTime", particlesAliveTimeCB);

        // Particles Velocities gpu buffer setting
        particlesVelCb = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesVelCb.SetData(particlesVel);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesVel", particlesVelCb);
        psScreenSpaceCollisionDetectionCs.SetBuffer(kernelIdScrSpaceColDetc, "particlesVel", particlesVelCb);
        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "particlesVel", particlesVelCb);

        // Particles without screen space collision detection gpu buffer setting
        particlesWithoutDepthCollisionCb = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3, ComputeBufferType.Append);
        particlesWithoutDepthCollisionCb.SetCounterValue(0);

        psScreenSpaceCollisionDetectionCs.SetBuffer(kernelIdScrSpaceColDetc, "particlesWithoutDepthCollision", particlesWithoutDepthCollisionCb);
        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "particlesWithoutDepthCollision", particlesWithoutDepthCollisionCb);
        fillBufferCs.SetBuffer(kernelIdFillBuffer, "buffer", particlesWithoutDepthCollisionCb);

        counterBuffer = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Raw);

        // Depth buffer for depth pre-pass setting
        depthTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.RFloat);
        depthTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        depthTexture.Create();

        psScreenSpaceCollisionDetectionCs.SetTexture(kernelIdScrSpaceColDetc, "depthTexture", depthTexture);

        // Normal buffer for normal pre-pass setting
        normalTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.ARGBFloat);
        normalTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        normalTexture.Create();

        psScreenSpaceCollisionDetectionCs.SetTexture(kernelIdScrSpaceColDetc, "normalTexture", normalTexture);

        Stopwatch sw0 = new Stopwatch();
        Stopwatch sw1 = new Stopwatch();

        sw0.Start();

        BuildBvhWithMortonCodes();
        sw1.Start();
        SplitLeafNodesWithSah();
        sw1.Stop();
        sw0.Stop();
        UnityEngine.Debug.Log("Time to build BVH with " + (numLastLevelBvh + 1) + " levels: " + sw0.Elapsed.TotalSeconds + " seconds");
        UnityEngine.Debug.Log("Time to compute SAH for " + (numLastLevelBvh - numLevelsBVHMorton + 1) + " levels: " + sw1.Elapsed.TotalSeconds + " seconds");
        PrintBvhNodes();
        UnityEngine.Debug.Log("Triangles after SAH: " + trisAfterSAH);

        // BVH structure gpu buffer setting
        bvhCb = new ComputeBuffer(bvh.Count, System.Runtime.InteropServices.Marshal.SizeOf(typeof(BvhSphereNodeGpu)), ComputeBufferType.Structured);

        BvhSphereNodeGpu[] bvhGpu = new BvhSphereNodeGpu[bvh.Count];

        for (int i = 0; i < bvh.Count; i++)
        {
            BvhSphereNode node = bvh[i];
            bvhGpu[i] = new BvhSphereNodeGpu(
                node.boundingSphere.center, 
                node.boundingSphere.radius,
                node.childrenORspan[0],
                node.childrenORspan[1]);
        }

        bvhCb.SetData(bvhGpu);

        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "bvh", bvhCb);

        // Stack for bvh nodes gpu buffer setting
        bvhStackCb = new ComputeBuffer(particlesPos.Count * bvhStackSizePerThread, sizeof(int), ComputeBufferType.Structured);

        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "bvhStack", bvhStackCb);

        // Stack indices of each thread gpu buffer setting
        bvhStackIndicesCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);

        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "bvhStackIndices", bvhStackIndicesCb);

        // BVH triangles gpu buffer setting
        bvhTrianglesCb = new ComputeBuffer(triangles.Count, System.Runtime.InteropServices.Marshal.SizeOf(typeof(BvhTriangleGpu)), ComputeBufferType.Structured);

        BvhTriangleGpu[] trianglesGpu = new BvhTriangleGpu[triangles.Count];

        for(int i = 0; i < triangles.Count; i++)
        {
            BvhTriangle tri = triangles[i];
            trianglesGpu[i] = new BvhTriangleGpu(tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        }

        bvhTrianglesCb.SetData(trianglesGpu);

        psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetc, "bvhTriangles", bvhTrianglesCb);

        commandBuf = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, commandCount, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[commandCount];
    }

    // Update is called once per frame
    void Update()
    {
        int threadGroupsX = Mathf.CeilToInt((float)particlesPos.Count / 32f);

        // Screen Space Particle Collision setting and dispatch
        DepthPrePass();
        NormalPrePass();
        textureImage.texture = depthTexture;

        psScreenSpaceCollisionDetectionCs.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        psScreenSpaceCollisionDetectionCs.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        psScreenSpaceCollisionDetectionCs.SetMatrix("inverseProjectionMat", Camera.main.projectionMatrix.inverse);
        psScreenSpaceCollisionDetectionCs.SetVector("cameraPos", Camera.main.transform.position);
        psScreenSpaceCollisionDetectionCs.SetFloat("particleRadius", particleRadius);
        psScreenSpaceCollisionDetectionCs.SetBool("isActive", isScreenSpaceCollisionActive);

        Vector2 screenRes = new(Screen.width, Screen.height);
        psScreenSpaceCollisionDetectionCs.SetVector("screenSize", screenRes);

        psScreenSpaceCollisionDetectionCs.Dispatch(kernelIdScrSpaceColDetc, threadGroupsX, 1, 1);

        // Filling the particlesWithoutDepthCollision buffer to align with the thread group size

        GraphicsBuffer.CopyCount(particlesWithoutDepthCollisionCb, counterBuffer, 0);

        int[] countArray = new int[1];
        counterBuffer.GetData(countArray);
        int numElementsToFill = threadGroupSize - (countArray[0] % threadGroupSize);
        if(numElementsToFill != threadGroupSize)
        {
            fillBufferCs.SetInt("numElements", numElementsToFill);

            fillBufferCs.Dispatch(kernelIdFillBuffer, 1, 1, 1);
        }

        // Volumes Structure Particle Collision setting and dispatch
        psVolumeStructureCollisionDetectionCs.SetFloat("particleRadius", particleRadius);
        psVolumeStructureCollisionDetectionCs.SetFloat("deltaTime", Time.deltaTime);
        psVolumeStructureCollisionDetectionCs.SetBool("isActive", isVolumeStructureCollisionActive);

        psVolumeStructureCollisionDetectionCs.Dispatch(kernelIdVolStructColDetc, threadGroupsX, 1, 1);

        // Partcle System reaction update setting and dispatch
        psReactionUpdateCs.SetFloat("deltaTime", Time.deltaTime);
        psReactionUpdateCs.Dispatch(kernelIdReactUpdate, threadGroupsX, 1, 1);

        // Particles mesh instancing rendering
        RenderParams rp = new RenderParams(instancedParticlesMat);
        rp.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
        rp.worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one);
        rp.matProps = new MaterialPropertyBlock();

        commandData[0].indexCountPerInstance = particleMesh.GetIndexCount(0);
        commandData[0].instanceCount = (uint)particlesPos.Count;
        commandBuf.SetData(commandData);
        Graphics.RenderMeshIndirect(rp, particleMesh, commandBuf, commandCount);

        particlesWithoutDepthCollisionCb.SetCounterValue(0);

        if (Input.GetKeyDown(KeyCode.Space))
        {
            bvhNodeLevelToRender = (bvhNodeLevelToRender + 1) % (numLastLevelBvh + 1);

            foreach (GameObject node in sphericalBvhNodes)
                Destroy(node);
            sphericalBvhNodes.Clear();

            int numNodes = (int)Mathf.Pow(2f, bvhNodeLevelToRender);

            for (int i = 0; i < numNodes; i++)
            {
                BvhSphereNode curNode = bvh[numNodes - 1 + i];
                sphericalBvhNodes.Add(Instantiate(sphericalNodePrefab));
                sphericalBvhNodes.Last().transform.position = curNode.boundingSphere.center;
                sphericalBvhNodes.Last().transform.localScale = Vector3.one * (curNode.boundingSphere.radius * 2f);
            }
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            foreach (GameObject node in sphericalBvhNodes)
                Destroy(node);
            sphericalBvhNodes.Clear();

            if (isRenderingLeaves)
            {
                isRenderingLeaves = false;
            }
            else
            {
                isRenderingLeaves = true;
                List<int> toTraverse = new() { 0 };

                while (toTraverse.Count > 0)
                {
                    BvhSphereNode curNode = bvh[toTraverse.Last()];
                    toTraverse.RemoveAt(toTraverse.Count - 1);

                    if (curNode.IsLeafNode())
                    {
                        sphericalBvhNodes.Add(Instantiate(sphericalNodePrefab));
                        sphericalBvhNodes.Last().transform.position = curNode.boundingSphere.center;
                        sphericalBvhNodes.Last().transform.localScale = Vector3.one * (curNode.boundingSphere.radius * 2f);
                    }
                    else
                    {
                        toTraverse.Add(curNode.childrenORspan[1]);
                        toTraverse.Add(curNode.childrenORspan[0]);
                    }
                }
            }
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            isScreenSpaceCollisionActive = !isScreenSpaceCollisionActive;
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            isVolumeStructureCollisionActive = !isVolumeStructureCollisionActive;
        }
    }

    void OnDestroy()
    {
        //BvhSphereNodeGpu[] bvhGpu = new BvhSphereNodeGpu[bvh.Count];
        //bvhCb.GetData(bvhGpu);

        //bool isEqual = true;
        //for (int i = 0; i < bvhGpu.Length; i++)
        //{
        //    if (!bvhGpu[i].Equals(bvh[i]))
        //    {
        //        isEqual = false;
        //        break;
        //    }
        //}

        //UnityEngine.Debug.Log("Bvh buffer is set right: " + isEqual);

        particlesPosCb?.Release();
        particlesPosCb = null;
        particlesVelCb?.Release();
        particlesVelCb = null;
        particlesWithoutDepthCollisionCb?.Release();
        particlesWithoutDepthCollisionCb = null;
        bvhCb?.Release();
        bvhCb = null;
        bvhStackCb?.Release();
        bvhStackCb = null;
        bvhStackIndicesCb?.Release();
        bvhStackIndicesCb = null;
        bvhTrianglesCb?.Release();
        bvhTrianglesCb = null;
        commandBuf?.Release();
        commandBuf = null;
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

    class BoundingSphere
    {
        public Vector3 center = Vector3.zero;
        public float radius = 0f;

        public static BoundingSphere Create(List<BvhTriangle> triangles)
        {
            BoundingSphere bs = new BoundingSphere();
            bs.center = Vector3.zero;
            bs.radius = 0f;
            int count = 0;
            foreach (BvhTriangle tri in triangles)
            {
                for (int v = 0; v < 3; v++)
                {
                    Vector3 point = tri.vertices[v];
                    bs.center += (point - bs.center) / (++count);
                }
            }

            BvhTriangle mostDistTri = new();
            int mostDistVert = 0;
            float maxDist = 0f;
            foreach (BvhTriangle tri in triangles)
            {
                for (int v = 0; v < 3; v++)
                {
                    Vector3 point = tri.vertices[v];

                    float curDist = Vector3.SqrMagnitude(point - bs.center);
                    if (curDist > maxDist)
                    {
                        mostDistTri = tri;
                        mostDistVert = v;
                        maxDist = curDist;
                    }
                }
            }

            bs.radius = Vector3.Distance(bs.center, mostDistTri.vertices[mostDistVert]) + 0.001f;

            return bs;
        }

        public float SufaceArea()
        {
            return 4f * Mathf.PI * radius * radius;
        }
    }

    class BvhTriangle : IComparable<BvhTriangle>
    {
        public Vector3[] vertices = new Vector3[3];
        public Vector3 centroid = Vector3.zero;
        public int mortonCode = 0;

        public int CompareTo(BvhTriangle other)
        {
            if(other == null) return 1;

            return mortonCode.CompareTo(other.mortonCode);
        }
    }

    [System.Serializable]
    struct BvhTriangleGpu
    {
        private Vector3 vert0;
        private Vector3 vert1;
        private Vector3 vert2;

        public BvhTriangleGpu(Vector3 v0, Vector3 v1, Vector3 v2)
        {
            vert0 = v0;
            vert1 = v1;
            vert2 = v2;
        }
    }

    class BvhSphereNode
    {
        public BoundingSphere boundingSphere = new();
        // If the node is a leaf node the first element is the index to the first triangle
        // of the scence's triangle list but negative and the second element is the number
        // of triangles encompassed by the bvh node. Otherwise the two elements are indices
        // to child nodes
        public int[] childrenORspan = new int[2] { 0, 0 };

        public static BvhSphereNode CreateNodeFromTriangles(List<BvhTriangle> triangles)
        {
            BvhSphereNode node = new()
            {
                boundingSphere = BoundingSphere.Create(triangles)
            };

            return node;
        }

        public bool IsLeafNode()
        {
            return childrenORspan[0] <= 0;
        }

        public int FirstTriIndex()
        {
            return Mathf.Abs(childrenORspan[0]);
        }

        public int LastTriIndexExclusive()
        {
            return Mathf.Abs(childrenORspan[0]) + childrenORspan[1];
        }

        public int TrisCount()
        {
            return childrenORspan[1];
        }
    }

    struct BvhSphereNodeGpu
    {
        private Vector3 center;
        private float radius;
        private int childOrStartNegated;
        private int childOrSize;

        public BvhSphereNodeGpu(Vector3 center, float radius, int childOrStartNegated, int childOrSize)
        {
            this.center = center;
            this.radius = radius;
            this.childOrStartNegated = childOrStartNegated;
            this.childOrSize = childOrSize;
        }

        public bool Equals(BvhSphereNode node)
        {
            return node.boundingSphere.center == center && 
                node.boundingSphere.radius == radius &&
                node.childrenORspan[0] == childOrStartNegated &&
                node.childrenORspan[1] == childOrSize;
        }
    };

    class TrianglesSpan
    {
        public List<BvhTriangle> triangles = new();
        public int firstElementIndex = -1;
    }
    
    private TrianglesSpan GetTrianglesInMortonCodeSpan(List<BvhTriangle> triangles, int minInclusive, int maxExclusive)
    {
        TrianglesSpan span = new TrianglesSpan();

        if (triangles.Count > 0) span.firstElementIndex = 0;

        foreach (BvhTriangle tri in triangles)
        {
            if(tri.mortonCode < minInclusive)
            {
                span.firstElementIndex++;
            }
            else if (tri.mortonCode >= minInclusive && tri.mortonCode < maxExclusive)
            {
                span.triangles.Add(tri);
            }
            else if (tri.mortonCode > maxExclusive)
            {
                break;
            }
        }

        return span;
    }

    void BuildBvhWithMortonCodes()
    {
        triangles = GetBvhTrianglesSortedWithMortonCodes();

        List<int> cutValues = new(){ 1 << 30 };

        for (int numLevels = 0; numLevels < numLevelsBVHMorton; numLevels++)
        {
            for (int i = 0; i < Mathf.Pow(2, numLevels); i++)
            {
                if (i == 0)
                {
                    TrianglesSpan triSpan = GetTrianglesInMortonCodeSpan(triangles, 0, cutValues[i]);
                    bvh.Add(BvhSphereNode.CreateNodeFromTriangles(triSpan.triangles));
                    bvh.Last().childrenORspan = new int[2] { -triSpan.firstElementIndex, triSpan.triangles.Count };
                }
                else
                {
                    TrianglesSpan triSpan = GetTrianglesInMortonCodeSpan(triangles, cutValues[i - 1], cutValues[i]);
                    bvh.Add(BvhSphereNode.CreateNodeFromTriangles(triSpan.triangles));
                    bvh.Last().childrenORspan = new int[2] { -triSpan.firstElementIndex, triSpan.triangles.Count };
                }
            }

            int newCutValue = cutValues.First() >> 1;
            int lastIndex = cutValues.Count - 1;
            for (int j = 0; j < lastIndex; j++)
            {
                cutValues.Add(cutValues[j] | newCutValue);
            }
            cutValues.Add(newCutValue);

            cutValues.Sort();
        }

        for (int i = 0; i < bvh.Count; i++)
        {
            int[] childrenIndices = new int[2] { 2 * i + 1, 2 * i + 2 };

            if (bvh.Count <= childrenIndices[1]) break;

            if (bvh[childrenIndices[0]].childrenORspan[1] > 0 &&
                bvh[childrenIndices[1]].childrenORspan[1] > 0)
            {
                bvh[i].childrenORspan = childrenIndices;
            }
        }
    }

    List<BvhTriangle> GetBvhTrianglesSortedWithMortonCodes()
    {
        BoundingBox box = new BoundingBox();

        List<BvhTriangle> triangles = new();

        // Find all GameObjects in the scene
        List<GameObject> allObjects = FindObjectsOfType<GameObject>().ToList();

        // Iterate through each GameObject and get its MeshFilter component
        foreach (GameObject obj in allObjects)
        {
            if (obj.TryGetComponent(out MeshFilter meshFilter))
            {
                Mesh mesh = meshFilter.sharedMesh;
                if (!mesh) continue;
                int vertexIndex = 3;
                foreach (int i in mesh.triangles)
                {
                    Vector3 vertex = obj.transform.TransformPoint(mesh.vertices[i]);
                    box.ScaleToInclude(vertex);

                    if (vertexIndex >= 3)
                    {
                        triangles.Add(new BvhTriangle());
                        vertexIndex = 0;
                    }
                    triangles.Last().vertices[vertexIndex++] = vertex;
                }
            }
        }

        foreach (BvhTriangle t in triangles)
        {
            t.centroid = (t.vertices[0] + t.vertices[1] + t.vertices[2]) / 3f;
        }

        box.center = (box.max + box.min) * 0.5f;
        box.length = (box.max - box.min);

        UnityEngine.Debug.Log("Number of triangles: " + triangles.Count);

        foreach (BvhTriangle tri in triangles)
        {
            tri.mortonCode = BoundingBox.TriangleMortonCode(tri.vertices.ToList(), box);
        }

        triangles.Sort();

        return triangles;
    }

    private void SplitLeafNodesWithSah(int maxTrisPerBVHNode = maxTrisPerBvhNode)
    {
        for (int nodeLevel = numLevelsBVHMorton; nodeLevel <= maxLevelBvh; nodeLevel++)
        {
            int levelNodeCount = Mathf.CeilToInt(Mathf.Pow(2f, nodeLevel));
            if (bvh.Count < levelNodeCount)
            {
                for (int i = 0; i < levelNodeCount; i++)
                {
                    bvh.Add(new BvhSphereNode());
                }
            }
        }

        List<int> nodesToTraverse = new List<int>() { 0 };

        while (nodesToTraverse.Count > 0)
        {
            int nodeIndex = nodesToTraverse.Last();
            nodesToTraverse.RemoveAt(nodesToTraverse.Count - 1);
            BvhSphereNode curNode = bvh[nodeIndex];

            if (curNode.IsLeafNode())
            {
                int nodeLevel = Mathf.FloorToInt(Mathf.Log(nodeIndex + 1f, 2f));
                numLastLevelBvh = Mathf.Max(numLastLevelBvh, nodeLevel);
                if (curNode.TrisCount() > maxTrisPerBVHNode && nodeLevel < maxLevelBvh)
                {
                    // Sample random triangle indices contained inside the current leaf node to evaluate SAH
                    List<int> trisSamples = new List<int>();

                    if (curNode.TrisCount() <= maxSahSamples)
                    {
                        for (int i = 0; i < curNode.TrisCount(); i++)
                        {
                            trisSamples.Add(curNode.FirstTriIndex() + i);
                        }
                    }
                    else
                    {
                        int bucketSize = Mathf.FloorToInt((float)curNode.TrisCount() / maxSahSamples);

                        for (int i = 0; i < maxSahSamples - 1; i++)
                        {
                            int randomOffset = Mathf.Max(Mathf.FloorToInt(UnityEngine.Random.Range(0, 1f) * bucketSize), bucketSize-1);
                            trisSamples.Add(curNode.FirstTriIndex() + Mathf.FloorToInt((bucketSize * i) + randomOffset));
                        }
                        {
                            int startIndexLastBucket = bucketSize * (maxSahSamples - 1);
                            int lastBucketSize = curNode.TrisCount() - startIndexLastBucket;
                            int randomOffset = Mathf.Max(Mathf.FloorToInt(UnityEngine.Random.Range(0, 1f) * lastBucketSize), lastBucketSize - 1);
                            trisSamples.Add(curNode.FirstTriIndex() + Mathf.FloorToInt(randomOffset + startIndexLastBucket));
                        }
                    }

                    // Determine split axis using SAH
                    int bestAxis = -1;
                    float bestPos = 0f, bestCost = float.MaxValue;
                    foreach (int indexSample in trisSamples)
                    {
                        BvhTriangle triangle = triangles[indexSample];
                        for (int axis = 0; axis < 3; axis++)
                        {
                            float candidatePos = triangle.centroid[axis];
                            float cost = EvaluateSah(curNode, axis, candidatePos);
                            if (cost < bestCost)
                            {
                                bestPos = candidatePos;
                                bestAxis = axis;
                                bestCost = cost;
                            }
                        }
                    }

                    int splitAxis = bestAxis;
                    float splitPos = bestPos;

                    int partIndex = Partition(triangles, splitAxis, splitPos, curNode.FirstTriIndex(), curNode.TrisCount());

                    int childIndex = 2 * nodeIndex + 1;

                    List<BvhTriangle> childTris = triangles.GetRange(curNode.FirstTriIndex(), partIndex - curNode.FirstTriIndex());
                    bvh[childIndex] = BvhSphereNode.CreateNodeFromTriangles(childTris);
                    bvh[childIndex].childrenORspan = new int[2] { -curNode.FirstTriIndex(), childTris.Count };

                    childIndex++;
                    childTris = triangles.GetRange(partIndex, curNode.TrisCount() - childTris.Count);
                    bvh[childIndex] = BvhSphereNode.CreateNodeFromTriangles(childTris);
                    bvh[childIndex].childrenORspan = new int[2] { -partIndex, childTris.Count };

                    curNode.childrenORspan = new int[2] { 2 * nodeIndex + 1, 2 * nodeIndex + 2 };

                    nodesToTraverse.Add(curNode.childrenORspan[1]);
                    nodesToTraverse.Add(curNode.childrenORspan[0]);
                }
            }
            else
            {
                nodesToTraverse.Add(curNode.childrenORspan[1]);
                nodesToTraverse.Add(curNode.childrenORspan[0]);
            }
        }
    }

    float EvaluateSah(BvhSphereNode node, int axis, float pos)
    {
        // determine triangle counts and bounds for this split candidate
        List<BvhTriangle> tris0 = new(), tris1 = new();
        int count0 = 0, count1 = 0;
        for (int i = node.FirstTriIndex(); i < node.LastTriIndexExclusive(); i++)
        {
            BvhTriangle tri = triangles[i];
            if (tri.centroid[axis] < pos)
            {
                count0++;
                tris0.Add(tri);
            }
            else
            {
                count1++;
                tris1.Add(tri);
            }
        }

        BoundingSphere bSphere0 = BoundingSphere.Create(tris0);
        BoundingSphere bSphere1 = BoundingSphere.Create(tris1);
        float cost = count0 * bSphere0.SufaceArea() + count1 * bSphere1.SufaceArea();
        return cost > 0f ? cost : float.MaxValue;
    }

    private static int Partition(List<BvhTriangle> tris, int axis, float pos, int startIndex, int count)
    {
        // Ensure valid subset range
        int endIndex = Mathf.Min(startIndex + count, tris.Count);
        if (startIndex >= tris.Count || count <= 0) return startIndex;

        int left = startIndex;
        int right = endIndex - 1;

        while (left <= right)
        {
            while (left <= right && tris[left].centroid[axis] < pos) left++;
            while (left <= right && tris[right].centroid[axis] >= pos) right--;
            if (left < right)
            {
                (tris[left], tris[right]) = (tris[right], tris[left]); // Swap
                left++;
                right--;
            }
        }

        return left; // Return partition index
    }

    private int trisAfterSAH = 0;

    private void PrintBvhNodes(int nodeIndex = 0, int nodeLevel = 0)
    {
        if (nodeIndex >= bvh.Count) return;

        string offset = "";

        for (int i = 0; i < nodeLevel; i++)
        {
            offset += "    ";
        }

        BvhSphereNode curNode = bvh[nodeIndex];

        if (curNode.childrenORspan[0] <= 0)
        {
            UnityEngine.Debug.Log(offset + nodeLevel + ". Center: " + curNode.boundingSphere.center + " Tris: " + curNode.TrisCount());
            trisAfterSAH += curNode.TrisCount();
        }
        else
        {
            UnityEngine.Debug.Log(offset + nodeLevel + ". Center: " + curNode.boundingSphere.center);
            PrintBvhNodes(2 * nodeIndex + 1, nodeLevel + 1);
            PrintBvhNodes(2 * nodeIndex + 2, nodeLevel + 1);
        }
    }

    private class Aabb
    {
        public Vector3 min = Vector3.positiveInfinity;
        public Vector3 max = Vector3.negativeInfinity;

        public void ScaleToInclude(Vector3 point)
        {
            min = Vector3.Min(min, point - Vector3.one * 0.01f);
            max = Vector3.Max(max, point + Vector3.one * 0.01f);
        }

        public void ScaleToInclude(BvhTriangle triangle)
        {
            foreach (Vector3 vertex in triangle.vertices)
            {
                ScaleToInclude(vertex);
            }
        }

        public static Aabb Create(List<BvhTriangle> triangles)
        {
            Aabb aabb = new Aabb();

            foreach (BvhTriangle tri in triangles)
            {
                aabb.ScaleToInclude(tri);
            }

            return aabb;
        }
    }

    class BvhAabbNode
    {
        public Aabb aabb = new();
        // If the node is a leaf node the first element is the index to the first triangle
        // of the scence's triangle list but negative and the second element is the number
        // of triangles encompassed by the bvh node. Otherwise the two elements are indices
        // to child nodes
        public int[] childrenORspan = new int[2] { 0, 0 };

        public static BvhAabbNode CreateNodeFromTriangles(List<BvhTriangle> triangles)
        {
            BvhAabbNode node = new()
            {
                aabb = Aabb.Create(triangles)
            };

            return node;
        }

        public bool IsLeafNode()
        {
            return childrenORspan[0] <= 0;
        }

        public int FirstTriIndex()
        {
            return Mathf.Abs(childrenORspan[0]);
        }

        public int LastTriIndexExclusive()
        {
            return Mathf.Abs(childrenORspan[0]) + childrenORspan[1];
        }

        public int TrisCount()
        {
            return childrenORspan[1];
        }
    }

    void BuildOctreeWithMortonCodes()
    {
        triangles = GetBvhTrianglesSortedWithMortonCodes();
    }
}
