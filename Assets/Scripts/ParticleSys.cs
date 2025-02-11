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

    private List<BVHTriangle> triangles = new List<BVHTriangle>();

    private List<BVHSphereNode> BVH = new List<BVHSphereNode>();
    private const int numLevelsBVHMorton = 6;
    private const int maxLevelBVH = 16;
    private const int maxTrisPerBVHNode = 32;
    private int numLastLevelBVH = 0;
    private const int maxSAHSamples = 64;

    [SerializeField]
    private GameObject sphericalNodePrefab;
    private List<GameObject> sphericalBVHNodes = new();
    private int BVHNodeLevelToRender = -1;
    private bool isRenderingLeaves = false;

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

        Stopwatch sw0 = new Stopwatch();
        Stopwatch sw1 = new Stopwatch();

        sw0.Start();

        BuildBVHWithMortonCodes();
        sw1.Start();
        SplitLeafNodesWithSAH();
        sw1.Stop();
        sw0.Stop();
        UnityEngine.Debug.Log("Time to build BVH with " + (numLastLevelBVH + 1) + " levels: " + sw0.Elapsed.TotalSeconds + " seconds");
        UnityEngine.Debug.Log("Time to compute SAH for " + (numLastLevelBVH - numLevelsBVHMorton + 1) + " levels: " + sw1.Elapsed.TotalSeconds + " seconds");
        PrintBVHNodes();
        UnityEngine.Debug.Log("Triangles after SAH: " + trisAfterSAH);
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

        if (Input.GetKeyDown(KeyCode.Space))
        {
            BVHNodeLevelToRender = (BVHNodeLevelToRender + 1) % (numLastLevelBVH + 1);

            foreach (GameObject node in sphericalBVHNodes)
                Destroy(node);
            sphericalBVHNodes.Clear();

            int numNodes = (int)Mathf.Pow(2f, BVHNodeLevelToRender);

            for (int i = 0; i < numNodes; i++)
            {
                BVHSphereNode curNode = BVH[numNodes - 1 + i];
                sphericalBVHNodes.Add(Instantiate(sphericalNodePrefab));
                sphericalBVHNodes.Last().transform.position = curNode.boundingSphere.center;
                sphericalBVHNodes.Last().transform.localScale = Vector3.one * (curNode.boundingSphere.radius * 2f);
            }
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            foreach (GameObject node in sphericalBVHNodes)
                Destroy(node);
            sphericalBVHNodes.Clear();

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
                    BVHSphereNode curNode = BVH[toTraverse.Last()];
                    toTraverse.RemoveAt(toTraverse.Count - 1);

                    if (curNode.IsLeafNode())
                    {
                        sphericalBVHNodes.Add(Instantiate(sphericalNodePrefab));
                        sphericalBVHNodes.Last().transform.position = curNode.boundingSphere.center;
                        sphericalBVHNodes.Last().transform.localScale = Vector3.one * (curNode.boundingSphere.radius * 2f);
                    }
                    else
                    {
                        toTraverse.Add(curNode.childrenORspan[1]);
                        toTraverse.Add(curNode.childrenORspan[0]);
                    }
                }
            }
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

    class BoundingSphere
    {
        public Vector3 center = Vector3.zero;
        public float radius = 0f;

        public static BoundingSphere Create(List<BVHTriangle> triangles)
        {
            BoundingSphere bs = new BoundingSphere();
            bs.center = Vector3.zero;
            bs.radius = 0f;
            int count = 0;
            foreach (BVHTriangle tri in triangles)
            {
                for (int v = 0; v < 3; v++)
                {
                    Vector3 point = tri.vertices[v];
                    bs.center += (point - bs.center) / (++count);
                }
            }

            BVHTriangle mostDistTri = new();
            int mostDistVert = 0;
            float maxDist = 0f;
            foreach (BVHTriangle tri in triangles)
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

    class BVHTriangle : IComparable<BVHTriangle>
    {
        public Vector3[] vertices = new Vector3[3];
        public Vector3 centroid = Vector3.zero;
        public int mortonCode = 0;

        public int CompareTo(BVHTriangle other)
        {
            if(other == null) return 1;

            return mortonCode.CompareTo(other.mortonCode);
        }
    }

    class BVHSphereNode
    {
        public BoundingSphere boundingSphere;
        // If the node is a leaf node the first element is the index to the first triangle
        // of the scence's triangle list but negative and the second element is the number
        // of triangles encompassed by the bvh node. Otherwise the two elements are indices
        // to child nodes
        public int[] childrenORspan = new int[2] { 0, 0 };

        public static BVHSphereNode CreateNodeFromTriangles(List<BVHTriangle> triangles)
        {
            BVHSphereNode node = new()
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

    class TrianglesSpan
    {
        public List<BVHTriangle> triangles = new();
        public int firstElementIndex = -1;
    }
    
    private TrianglesSpan GetTrianglesInMortonCodeSpan(List<BVHTriangle> triangles, int minInclusive, int maxExclusive)
    {
        TrianglesSpan span = new TrianglesSpan();

        if (triangles.Count > 0) span.firstElementIndex = 0;

        foreach (BVHTriangle tri in triangles)
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

    void BuildBVHWithMortonCodes()
    {
        BoundingBox box = new BoundingBox();

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
                        triangles.Add(new BVHTriangle());
                        vertexIndex = 0;
                    }
                    triangles.Last().vertices[vertexIndex++] = vertex;
                }
            }
        }

        foreach(BVHTriangle t in triangles)
        {
            t.centroid = (t.vertices[0] + t.vertices[1] + t.vertices[2]) / 3f;
        }

        box.center = (box.max + box.min) * 0.5f;
        box.length = (box.max - box.min);

        UnityEngine.Debug.Log("Number of triangles: " +  triangles.Count);

        foreach(BVHTriangle tri in triangles)
        {
            tri.mortonCode = BoundingBox.TriangleMortonCode(tri.vertices.ToList(), box);
        }

        triangles.Sort();

        List<int> cutValues = new(){ 1 << 30 };

        for (int numLevels = 0; numLevels < numLevelsBVHMorton; numLevels++)
        {
            for (int i = 0; i < Mathf.Pow(2, numLevels); i++)
            {
                if (i == 0)
                {
                    TrianglesSpan triSpan = GetTrianglesInMortonCodeSpan(triangles, 0, cutValues[i]);
                    BVH.Add(BVHSphereNode.CreateNodeFromTriangles(triSpan.triangles));
                    BVH.Last().childrenORspan = new int[2] { -triSpan.firstElementIndex, triSpan.triangles.Count };
                }
                else
                {
                    TrianglesSpan triSpan = GetTrianglesInMortonCodeSpan(triangles, cutValues[i - 1], cutValues[i]);
                    BVH.Add(BVHSphereNode.CreateNodeFromTriangles(triSpan.triangles));
                    BVH.Last().childrenORspan = new int[2] { -triSpan.firstElementIndex, triSpan.triangles.Count };
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

        for (int i = 0; i < BVH.Count; i++)
        {
            int[] childrenIndices = new int[2] { 2 * i + 1, 2 * i + 2 };

            if (BVH.Count <= childrenIndices[1]) break;

            if (BVH[childrenIndices[0]].childrenORspan[1] > 0 &&
                BVH[childrenIndices[1]].childrenORspan[1] > 0)
            {
                BVH[i].childrenORspan = childrenIndices;
            }
        }
    }

    private void SplitLeafNodesWithSAH(int maxTrisPerBVHNode = maxTrisPerBVHNode)
    {
        for (int nodeLevel = numLevelsBVHMorton; nodeLevel <= maxLevelBVH; nodeLevel++)
        {
            int levelNodeCount = Mathf.CeilToInt(Mathf.Pow(2f, nodeLevel));
            if (BVH.Count < levelNodeCount)
            {
                for (int i = 0; i < levelNodeCount; i++)
                {
                    BVH.Add(new BVHSphereNode());
                }
            }
        }

        List<int> nodesToTraverse = new List<int>() { 0 };

        while (nodesToTraverse.Count > 0)
        {
            int nodeIndex = nodesToTraverse.Last();
            nodesToTraverse.RemoveAt(nodesToTraverse.Count - 1);
            BVHSphereNode curNode = BVH[nodeIndex];

            if (curNode.IsLeafNode())
            {
                int nodeLevel = Mathf.FloorToInt(Mathf.Log(nodeIndex + 1f, 2f));
                numLastLevelBVH = Mathf.Max(numLastLevelBVH, nodeLevel);
                if (curNode.TrisCount() > maxTrisPerBVHNode && nodeLevel < maxLevelBVH)
                {
                    // Sample random triangle indices contained inside the current leaf node to evaluate SAH
                    List<int> trisSamples = new List<int>();

                    if (curNode.TrisCount() <= maxSAHSamples)
                    {
                        for (int i = 0; i < curNode.TrisCount(); i++)
                        {
                            trisSamples.Add(curNode.FirstTriIndex() + i);
                        }
                    }
                    else
                    {
                        int bucketSize = Mathf.FloorToInt((float)curNode.TrisCount() / maxSAHSamples);

                        for (int i = 0; i < maxSAHSamples - 1; i++)
                        {
                            int randomOffset = Mathf.Max(Mathf.FloorToInt(UnityEngine.Random.Range(0, 1f) * bucketSize), bucketSize-1);
                            trisSamples.Add(curNode.FirstTriIndex() + Mathf.FloorToInt((bucketSize * i) + randomOffset));
                        }
                        {
                            int startIndexLastBucket = bucketSize * (maxSAHSamples - 1);
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
                        BVHTriangle triangle = triangles[indexSample];
                        for (int axis = 0; axis < 3; axis++)
                        {
                            float candidatePos = triangle.centroid[axis];
                            float cost = EvaluateSAH(curNode, axis, candidatePos);
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

                    List<BVHTriangle> childTris = triangles.GetRange(curNode.FirstTriIndex(), partIndex - curNode.FirstTriIndex());
                    BVH[childIndex] = BVHSphereNode.CreateNodeFromTriangles(childTris);
                    BVH[childIndex].childrenORspan = new int[2] { -curNode.FirstTriIndex(), childTris.Count };

                    childIndex++;
                    childTris = triangles.GetRange(partIndex, curNode.TrisCount() - childTris.Count);
                    BVH[childIndex] = BVHSphereNode.CreateNodeFromTriangles(childTris);
                    BVH[childIndex].childrenORspan = new int[2] { -partIndex, childTris.Count };

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

    float EvaluateSAH(BVHSphereNode node, int axis, float pos)
    {
        // determine triangle counts and bounds for this split candidate
        List<BVHTriangle> tris0 = new(), tris1 = new();
        int count0 = 0, count1 = 0;
        for (int i = node.FirstTriIndex(); i < node.LastTriIndexExclusive(); i++)
        {
            BVHTriangle tri = triangles[i];
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

    private static int Partition(List<BVHTriangle> tris, int axis, float pos, int startIndex, int count)
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

    private void PrintBVHNodes(int nodeIndex = 0, int nodeLevel = 0)
    {
        if (nodeIndex >= BVH.Count) return;

        string offset = "";

        for (int i = 0; i < nodeLevel; i++)
        {
            offset += "    ";
        }

        BVHSphereNode curNode = BVH[nodeIndex];

        if (curNode.childrenORspan[0] <= 0)
        {
            UnityEngine.Debug.Log(offset + nodeLevel + ". Center: " + curNode.boundingSphere.center + " Tris: " + curNode.TrisCount());
            trisAfterSAH += curNode.TrisCount();
        }
        else
        {
            UnityEngine.Debug.Log(offset + nodeLevel + ". Center: " + curNode.boundingSphere.center);
            PrintBVHNodes(2 * nodeIndex + 1, nodeLevel + 1);
            PrintBVHNodes(2 * nodeIndex + 2, nodeLevel + 1);
        }
    }
}
