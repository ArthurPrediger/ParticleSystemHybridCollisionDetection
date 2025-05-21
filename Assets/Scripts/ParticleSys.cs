#define PERFORMANCE_BENCHMARK
//#define ACCURACY_BENCHMARK

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class ParticleSys : MonoBehaviour
{
    [SerializeField]
    private ComputeShader psReactionUpdateCs;
    private int kernelIdReactUpdate;
    [SerializeField]
    private ComputeShader screenSpaceDepthCollisionDetectionCs;
    private int kernelIdScrSpcDepthColDetc;
    private int kernelIdScrSpcDepthColDetcHybrid;
    [SerializeField]
    private ComputeShader spatialStructureCollisionDetectionCs;
    private int kernelIdSptStructColDetc;
    private int kernelIdSptStructColDetcHybrid;
    [SerializeField]
    private ComputeShader computeDispatchArgsCs;
    private int kernelIdCompDispArgs;

    [SerializeField]
    private Shader DepthPrePassShader;
    [SerializeField]
    private Shader NormalPrePassShader;

    [SerializeField]
    private Material instancedParticlesMat;
    [SerializeField]
    private Mesh particleMesh;
    public float particleRadius = 2f;
    public int particlesLifetimeSteps = 2001;
    public int numParticlesXZ = 128;
    public float particlesOffsetXZ = 4f;
    public float deltaTime = 0.01f;
    public float particleBounciness = 0.25f;

    private List<Vector3> particlesPos = new();
    private List<Vector3> particlesVel = new();
    private List<int> particlesAliveTime = new();
    //private List<float> particlesLifeSpan = new();

    private ComputeBuffer particlesPosCb;
    private ComputeBuffer particlesVelCb;
    private ComputeBuffer particlesInitPosCB;
    private ComputeBuffer particlesAliveTimeCB;
    private ComputeBuffer particlesWithoutDepthCollisionCb;
    private ComputeBuffer numParticlesWithoutDepthCollisionCb;
    private ComputeBuffer argsBufferCb;
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
    //private const int numLevelsBVHMorton = 14;
    //private const int maxLevelBvh = 16;
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

    private const int threadGroupSize = 32;
    private const int bvhStackSizePerThread = 128;
    int threadGroupsX;

    private bool isScreenSpaceCollisionActive = true;
    private bool isVolumeStructureCollisionActive = false;

    private GraphicsBuffer commandBuf;
    private GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    private RenderParams rp;
    private const int commandCount = 1;

    private const float infinityFloatGpu = 1.0e38f;
    private Vector3 gravity = new(0f, -9.81f, 0f);

    private Vector2Int lastScreenSize = new();

    //private Stopwatch benchmarkSw = new();
#if PERFORMANCE_BENCHMARK
    private List<float> benchmarkTimingsScrSpace;
    private List<float> benchmarkTimingsVolStrc;
    private List<float> benchmarkTimingsHybrid;
#endif

#if ACCURACY_BENCHMARK
    private ComputeBuffer numCollisionsScrSpaceDepthCb;
    private ComputeBuffer numCollisionsVolStructureCb;
    private ComputeBuffer numCollisionsHybridCb;
    //private ComputeBuffer numCollisionsHybridVolCb;
#endif

    // Start is called before the first frame update
    void Start()
    {
        instancedParticlesMat.enableInstancing = true;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        GetComponent<MeshRenderer>().enabled = false;
        GetComponent<MeshFilter>().mesh = null;

        lastScreenSize = new Vector2Int(Screen.width, Screen.height);

#if PERFORMANCE_BENCHMARK
        BenchmarkManager bm = GetComponent<BenchmarkManager>();
        if(bm)
        {
            int benchTimingsCount = particlesLifetimeSteps * bm.GetCamerasCount();
            benchmarkTimingsScrSpace = new(benchTimingsCount);
            benchmarkTimingsVolStrc = new(benchTimingsCount);
            benchmarkTimingsHybrid = new(benchTimingsCount);
        }
#endif

        //SetupParticleSystemData(1);
        //VisualizeAllBvhNodes();
        //enabled = false;
    }

    public void SetupParticleSystemData(int particleLayersY)
    {
        int xzDimension = numParticlesXZ;

        if (particlesPos.Count == xzDimension * xzDimension * particleLayersY) return;

        // Compute buffer IDs setting
        kernelIdReactUpdate = psReactionUpdateCs.FindKernel("PSReactionUpdate");
        kernelIdScrSpcDepthColDetc = screenSpaceDepthCollisionDetectionCs.FindKernel("ScreenSpaceDepthCollisionDetection");
        kernelIdScrSpcDepthColDetcHybrid = screenSpaceDepthCollisionDetectionCs.FindKernel("ScreenSpaceDepthCollisionDetectionHybrid");
        kernelIdSptStructColDetc = spatialStructureCollisionDetectionCs.FindKernel("SpatialStructureCollisionDetection");
        kernelIdSptStructColDetcHybrid = spatialStructureCollisionDetectionCs.FindKernel("SpatialStructureCollisionDetectionHybrid");
        kernelIdCompDispArgs = computeDispatchArgsCs.FindKernel("ComputeDispatchArgs");

        // CPU and GPU particle data buffers setup
        SetupParticleDependentData(xzDimension, particleLayersY);

        if (bvh.Count > 0) return;

        // Num particles without screen space collision detection gpu buffer setting
        numParticlesWithoutDepthCollisionCb = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
        uint[] zero = new uint[1] { 0 };
        numParticlesWithoutDepthCollisionCb.SetData(zero);

        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetcHybrid, "numParticlesWithoutDepthCollision", numParticlesWithoutDepthCollisionCb);
        computeDispatchArgsCs.SetBuffer(kernelIdCompDispArgs, "numParticlesWithoutDepthCollision", numParticlesWithoutDepthCollisionCb);

        // Dispatch args gpu buffer setting
        argsBufferCb = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);
        argsBufferCb.SetData(new uint[3] { 0, 0, 0 });

        computeDispatchArgsCs.SetBuffer(kernelIdCompDispArgs, "dispatchArgs", argsBufferCb);

        // Depth and Normal textures for the Screen Space Collision Dectection method setup
        SetupDepthAndNormalPrePassBuffers();

        // BVH building and gpu BVH buffers setting up
        BuildAndSetupBvh();
    }

    private void SetupParticleDependentData(int xzDimension, int yDimension)
    {
        particlesPos.Clear();
        particlesPos.TrimExcess();
        particlesVel.Clear();
        particlesPos.TrimExcess();
        particlesAliveTime.Clear();
        particlesAliveTime.TrimExcess();

        particlesPosCb?.Release();
        particlesPosCb = null;
        particlesVelCb?.Release();
        particlesVelCb = null;
        particlesWithoutDepthCollisionCb?.Release();
        particlesWithoutDepthCollisionCb = null;
        particlesInitPosCB?.Release();
        particlesInitPosCB = null;
        particlesAliveTimeCB?.Release();
        particlesAliveTimeCB = null;
        bvhStackCb?.Release();
        bvhStackCb = null;
        bvhStackIndicesCb?.Release();
        bvhStackIndicesCb = null;
        commandBuf?.Release();
        commandBuf = null;
        commandData = null;

        // Initialization of particles positions and velocities
        float xzStart = (float)(xzDimension - 1) / 2f;
        float offset = particlesOffsetXZ;
        Vector3 starPos = new Vector3(xzStart, 0f, xzStart) * offset + transform.position;
        for (int i = 0; i < xzDimension; i++)
        {
            for (int j = 0; j < yDimension; j++)
            {
                for (int k = 0; k < xzDimension; k++)
                {
                    if (particlesPos.Count >= 65535 * 32) break;

                    particlesPos.Add((starPos - new Vector3(offset * i, -(offset * j * 4), offset * k)));
                    particlesVel.Add(Vector3.zero);
                    particlesAliveTime.Add(0);
                }
            }
        }

        threadGroupsX = Mathf.CeilToInt((float)particlesPos.Count / (float)threadGroupSize);

        if(particlesPos.Count % threadGroupSize != 0)
        {
            for (int i = particlesPos.Count % threadGroupSize; i < threadGroupSize; i++)
            {
                particlesPos.Add(Vector3.one * infinityFloatGpu);
                particlesVel.Add(Vector3.zero);
                particlesAliveTime.Add(0);
            }
        }

        // Particles Positions gpu buffer setting
        particlesPosCb = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesPosCb.SetData(particlesPos);

        instancedParticlesMat.SetBuffer("particlesPos", particlesPosCb);
        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesPos", particlesPosCb);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetc, "particlesPos", particlesPosCb);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetcHybrid, "particlesPos", particlesPosCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "particlesPos", particlesPosCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "particlesPos", particlesPosCb);

        // Particles Initial Positions gpu buffer setting
        particlesInitPosCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesInitPosCB.SetData(particlesPos);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesInitPos", particlesInitPosCB);

        // Particles Alive Time gpu buffer setting
        particlesAliveTimeCB = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        particlesAliveTimeCB.SetData(particlesAliveTime);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesAliveTime", particlesAliveTimeCB);

        // Particles Velocities gpu buffer setting
        particlesVelCb = new ComputeBuffer(particlesVel.Count, sizeof(float) * 3, ComputeBufferType.Structured);
        particlesVelCb.SetData(particlesVel);

        psReactionUpdateCs.SetBuffer(kernelIdReactUpdate, "particlesVel", particlesVelCb);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetc, "particlesVel", particlesVelCb);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetcHybrid, "particlesVel", particlesVelCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "particlesVel", particlesVelCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "particlesVel", particlesVelCb);

        // Particles without screen space collision detection gpu buffer setting
        particlesWithoutDepthCollisionCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        int[] zeros = new int[particlesPos.Count];
        for (int i = 0; i < zeros.Length; i++)
            zeros[i] = 0;
        particlesWithoutDepthCollisionCb.SetData(zeros);

        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetcHybrid, "particlesWithoutDepthCollision", particlesWithoutDepthCollisionCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "particlesWithoutDepthCollision", particlesWithoutDepthCollisionCb);

        // Stack for bvh nodes gpu buffer setting
        bvhStackCb = new ComputeBuffer(particlesPos.Count * bvhStackSizePerThread, sizeof(int), ComputeBufferType.Structured);

        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "bvhStack", bvhStackCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "bvhStack", bvhStackCb);

        // Stack indices of each thread gpu buffer setting
        bvhStackIndicesCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);

        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "bvhStackIndices", bvhStackIndicesCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "bvhStackIndices", bvhStackIndicesCb);

        // Initialization of data for mesh instancing rendering of the particles
        commandBuf = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, commandCount, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[commandCount];

        rp = new RenderParams(instancedParticlesMat);
        rp.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        rp.worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one);
        rp.matProps = new MaterialPropertyBlock();
        rp.matProps.SetFloat("particleRadius", particleRadius);

        commandData[0].indexCountPerInstance = particleMesh.GetIndexCount(0);
        commandData[0].instanceCount = (uint)particlesPos.Count;
        commandBuf.SetData(commandData);

#if ACCURACY_BENCHMARK
        numCollisionsScrSpaceDepthCb?.Release();
        numCollisionsScrSpaceDepthCb = null;
        numCollisionsVolStructureCb?.Release();
        numCollisionsVolStructureCb = null;
        numCollisionsHybridCb?.Release();
        numCollisionsHybridCb = null;
        //numCollisionsHybridVolCb?.Release();
        //numCollisionsHybridVolCb = null;

        List<int> numCollisionsZeroed = new();
        for (int i = 0; i < particlesPos.Count; i++)
        {
            numCollisionsZeroed.Add(0);
        }

        numCollisionsScrSpaceDepthCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        numCollisionsScrSpaceDepthCb.SetData(numCollisionsZeroed);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetc, "numCollisions", numCollisionsScrSpaceDepthCb);

        numCollisionsVolStructureCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        numCollisionsVolStructureCb.SetData(numCollisionsZeroed);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "numCollisions", numCollisionsVolStructureCb);

        numCollisionsHybridCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        numCollisionsHybridCb.SetData(numCollisionsZeroed);
        screenSpaceDepthCollisionDetectionCs.SetBuffer(kernelIdScrSpcDepthColDetcHybrid, "numCollisions", numCollisionsHybridCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "numCollisions", numCollisionsHybridCb);

        //numCollisionsHybridVolCb = new ComputeBuffer(particlesPos.Count, sizeof(int), ComputeBufferType.Structured);
        //numCollisionsHybridVolCb.SetData(numCollisionsZeroed);
        //psVolumeStructureCollisionDetectionCs.SetBuffer(kernelIdVolStructColDetcHybrid, "numCollisions", numCollisionsHybridVolCb);
#endif
    }

    private void SetupDepthAndNormalPrePassBuffers()
    {
        if (depthTexture) depthTexture.Release();
        if (normalTexture) normalTexture.Release();

        // Depth buffer for depth pre-pass setting
        depthTexture = new RenderTexture(Screen.width, Screen.height, 1, RenderTextureFormat.RFloat);
        depthTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        depthTexture.Create();

        screenSpaceDepthCollisionDetectionCs.SetTexture(kernelIdScrSpcDepthColDetc, "depthTexture", depthTexture);
        screenSpaceDepthCollisionDetectionCs.SetTexture(kernelIdScrSpcDepthColDetcHybrid, "depthTexture", depthTexture);

        // Normal buffer for normal pre-pass setting
        normalTexture = new RenderTexture(Screen.width, Screen.height, 1, RenderTextureFormat.ARGBFloat);
        normalTexture.enableRandomWrite = true;  // Enable random write for compute shader access
        normalTexture.Create();

        screenSpaceDepthCollisionDetectionCs.SetTexture(kernelIdScrSpcDepthColDetc, "normalTexture", normalTexture);
        screenSpaceDepthCollisionDetectionCs.SetTexture(kernelIdScrSpcDepthColDetcHybrid, "normalTexture", normalTexture);
    }

    private void BuildAndSetupBvh()
    {
        // BVH building for the current scene
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

        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "bvh", bvhCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "bvh", bvhCb);

        // BVH triangles gpu buffer setting
        bvhTrianglesCb = new ComputeBuffer(triangles.Count, System.Runtime.InteropServices.Marshal.SizeOf(typeof(BvhTriangleGpu)), ComputeBufferType.Structured);

        BvhTriangleGpu[] trianglesGpu = new BvhTriangleGpu[triangles.Count];

        for (int i = 0; i < triangles.Count; i++)
        {
            BvhTriangle tri = triangles[i];
            trianglesGpu[i] = new BvhTriangleGpu(tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        }

        bvhTrianglesCb.SetData(trianglesGpu);

        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetc, "bvhTriangles", bvhTrianglesCb);
        spatialStructureCollisionDetectionCs.SetBuffer(kernelIdSptStructColDetcHybrid, "bvhTriangles", bvhTrianglesCb);
    }

    // Update is called once per frame
    void Update()
    {
        // Screen Space Particle Collision setting and dispatch
        if (IsScreenSpaceCollisionActive())
        {
            RunScreenSpaceCollisionDetection(kernelIdScrSpcDepthColDetc);
#if PERFORMANCE_BENCHMARK
            benchmarkTimingsScrSpace.Add(Time.deltaTime * 1000f);
#endif
        }

        // Volumes Structure Particle Collision setting and dispatch
        if (IsVolumeStructureCollisionActive())
        {
            RunVolumeStructureCollisionDetection();
#if PERFORMANCE_BENCHMARK
            benchmarkTimingsVolStrc.Add(Time.deltaTime * 1000f);
#endif
        }

        // Screen Space and Volumes Structure Particle Collision Hybrid Method setting and dispatch
        if (IsHybridCollisionActive())
        {
            RunHybridCollisionDetection();
#if PERFORMANCE_BENCHMARK
            benchmarkTimingsHybrid.Add(Time.deltaTime * 1000f);
#endif
        }

        // Particle System reaction update setting and dispatch
        psReactionUpdateCs.SetVector("gravity", gravity);
        psReactionUpdateCs.SetFloat("deltaTime", deltaTime);
        psReactionUpdateCs.SetInt("particlesLifetimeSteps", particlesLifetimeSteps);
        psReactionUpdateCs.Dispatch(kernelIdReactUpdate, threadGroupsX, 1, 1);

        // Particles mesh instancing rendering
        Graphics.RenderMeshIndirect(rp, particleMesh, commandBuf, commandCount);

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
        particlesPosCb?.Release();
        particlesPosCb = null;
        particlesVelCb?.Release();
        particlesVelCb = null;
        particlesInitPosCB?.Release();
        particlesInitPosCB = null;
        particlesAliveTimeCB?.Release();
        particlesAliveTimeCB = null;
        particlesWithoutDepthCollisionCb?.Release();
        particlesWithoutDepthCollisionCb = null;
        numParticlesWithoutDepthCollisionCb?.Release();
        numParticlesWithoutDepthCollisionCb = null;
        argsBufferCb?.Release();
        argsBufferCb = null;
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
        commandData = null;
        if (depthTexture) depthTexture.Release();
        if(normalTexture) normalTexture.Release();

#if ACCURACY_BENCHMARK
        numCollisionsScrSpaceDepthCb?.Release();
        numCollisionsScrSpaceDepthCb = null;
        numCollisionsVolStructureCb?.Release();
        numCollisionsVolStructureCb = null;
        numCollisionsHybridCb?.Release();
        numCollisionsHybridCb = null;
        //numCollisionsHybridVolCb?.Release();
        //numCollisionsHybridVolCb = null;
#endif
    }

    void RunScreenSpaceCollisionDetection(int kernelId)
    {
        Vector2Int curScreenSize = new Vector2Int(Screen.width, Screen.height);

        if(curScreenSize != lastScreenSize)
        {
            lastScreenSize = curScreenSize;
            SetupDepthAndNormalPrePassBuffers();
        }

        DepthPrePass();
        NormalPrePass();
        //textureImage.texture = depthTexture;

        screenSpaceDepthCollisionDetectionCs.SetMatrix("projectionMat", Camera.main.projectionMatrix);
        screenSpaceDepthCollisionDetectionCs.SetMatrix("viewMat", Camera.main.worldToCameraMatrix);
        screenSpaceDepthCollisionDetectionCs.SetVector("cameraPos", Camera.main.transform.position);
        screenSpaceDepthCollisionDetectionCs.SetVector("cameraForward", Camera.main.transform.forward);
        screenSpaceDepthCollisionDetectionCs.SetVector("gravity", gravity);
        screenSpaceDepthCollisionDetectionCs.SetFloat("particleRadius", particleRadius);
        screenSpaceDepthCollisionDetectionCs.SetFloat("particleBounciness", particleBounciness);
        screenSpaceDepthCollisionDetectionCs.SetFloat("deltaTime", deltaTime);

        Vector2 screenRes = new(Screen.width, Screen.height);
        screenSpaceDepthCollisionDetectionCs.SetVector("screenSize", screenRes);

        screenSpaceDepthCollisionDetectionCs.Dispatch(kernelId, threadGroupsX, 1, 1);
    }

    void RunVolumeStructureCollisionDetection()
    {
        spatialStructureCollisionDetectionCs.SetVector("gravity", gravity);
        spatialStructureCollisionDetectionCs.SetFloat("particleRadius", particleRadius);
        spatialStructureCollisionDetectionCs.SetFloat("particleBounciness", particleBounciness);
        spatialStructureCollisionDetectionCs.SetFloat("deltaTime", deltaTime);
        spatialStructureCollisionDetectionCs.SetInt("maxStackSize", bvhStackSizePerThread);

        spatialStructureCollisionDetectionCs.Dispatch(kernelIdSptStructColDetc, threadGroupsX, 1, 1);
    }

    void RunHybridCollisionDetection()
    {
        // Screen Space Particle Collision setting and dispatch
        RunScreenSpaceCollisionDetection(kernelIdScrSpcDepthColDetcHybrid);

        //Finding the number of particles that do not complete a thread group
        computeDispatchArgsCs.SetInt("threadGroupSize", threadGroupSize);
        computeDispatchArgsCs.Dispatch(kernelIdCompDispArgs, 1, 1, 1);

        // Volumes Structure Particle Collision setting and dispatch
        spatialStructureCollisionDetectionCs.SetVector("gravity", gravity);
        spatialStructureCollisionDetectionCs.SetFloat("particleRadius", particleRadius);
        spatialStructureCollisionDetectionCs.SetFloat("particleBounciness", particleBounciness);
        spatialStructureCollisionDetectionCs.SetFloat("deltaTime", deltaTime);
        spatialStructureCollisionDetectionCs.SetInt("maxStackSize", bvhStackSizePerThread);

        spatialStructureCollisionDetectionCs.DispatchIndirect(kernelIdSptStructColDetcHybrid, argsBufferCb, 0);
    }

    void DepthPrePass()
    {
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.targetTexture = depthTexture;

            mainCamera.RenderWithShader(DepthPrePassShader, null);

            mainCamera.targetTexture = null;
        }
    }

    void NormalPrePass()
    {
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.targetTexture = normalTexture;

            mainCamera.RenderWithShader(NormalPrePassShader, null);

            mainCamera.targetTexture = null;
        }
    }

    public void SetScreenSpaceCollisionActive()
    {
        isScreenSpaceCollisionActive = true;
        isVolumeStructureCollisionActive = false;
    }

    public bool IsScreenSpaceCollisionActive()
    {
        return isScreenSpaceCollisionActive && !isVolumeStructureCollisionActive;
    }

    public void SetVolumeStructureCollisionActive()
    {
        isScreenSpaceCollisionActive = false;
        isVolumeStructureCollisionActive = true;
    }

    public bool IsVolumeStructureCollisionActive()
    {
        return !isScreenSpaceCollisionActive && isVolumeStructureCollisionActive;
    }

    public void SetHybridCollisionActive()
    {
        isScreenSpaceCollisionActive = true;
        isVolumeStructureCollisionActive = true;
    }

    public bool IsHybridCollisionActive()
    {
        return isScreenSpaceCollisionActive && isVolumeStructureCollisionActive;
    }

    public List<string> GetCollisionDetectionMethodsNames()
    {
        return new() {
            "Screen Space Collision Detection",
            "Volume Structure Collision Detection",
            "Hybrid Depth Collision Detection",
            //"Hybrid Volume Collision Detection"
        };
    }

#if PERFORMANCE_BENCHMARK
    public List<List<float>> GetBenchmarkTimings()
    {
        return new() {
            benchmarkTimingsScrSpace,
            benchmarkTimingsVolStrc,
            benchmarkTimingsHybrid,
        };
    }

    public void ResetBenchmarkTimings()
    {
        benchmarkTimingsScrSpace?.Clear();
        benchmarkTimingsVolStrc?.Clear();
        benchmarkTimingsHybrid?.Clear();
    }
#endif

#if ACCURACY_BENCHMARK
    public List<int[]> GetBenchmarkCollisions()
    {
        int[] numCollisionsScrSpaceDepth = new int[particlesPos.Count];
        numCollisionsScrSpaceDepthCb?.GetData(numCollisionsScrSpaceDepth);
        int[] numCollisionsVolStructure = new int[particlesPos.Count];
        numCollisionsVolStructureCb?.GetData(numCollisionsVolStructure);
        int[] numCollisionsHybrid = new int[particlesPos.Count];
        numCollisionsHybridCb?.GetData(numCollisionsHybrid);
        //int[] numCollisionsHybridVol = new int[particlesPos.Count];
        //numCollisionsHybridVolCb?.GetData(numCollisionsHybridVol);

        return new() { 
            numCollisionsScrSpaceDepth, 
            numCollisionsVolStructure, 
            numCollisionsHybrid,
            //numCollisionsHybridVol
        };
    }

    public void ResetBenchmarkCollisons()
    {
        List<int> numCollisionsZeroed = new(particlesPos.Count);
        for (int i = 0; i < particlesPos.Count; i++)
        {
            numCollisionsZeroed.Add(0);
        }

        numCollisionsScrSpaceDepthCb?.SetData(numCollisionsZeroed);
        numCollisionsVolStructureCb?.SetData(numCollisionsZeroed);
        numCollisionsHybridCb?.SetData(numCollisionsZeroed);
        //numCollisionsHybridVolCb?.SetData(numCollisionsZeroed);
    }
#endif

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

    List<BvhTriangle> GetBvhTrianglesSortedWithMortonCodes()
    {
        BoundingBox box = new BoundingBox();

        List<BvhTriangle> triangles = new();

        // Find all GameObjects in the scene
        List<GameObject> allObjects = FindObjectsOfType<GameObject>().ToList();

        // Iterate through each GameObject and get its MeshFilter component
        foreach (GameObject obj in allObjects)
        {
            if (obj == null || obj == this.gameObject) continue;
            if (obj.TryGetComponent(out MeshFilter meshFilter))
            {
                Mesh mesh = meshFilter.sharedMesh;
                if (!mesh) continue;
                int vertexIndex = 3;
                int[] tris = mesh.triangles;
                Vector3[] vertices = mesh.vertices;
                foreach (int i in tris)
                {
                    Vector3 vertex = obj.transform.TransformPoint(vertices[i]);
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

        int numBvhNodes = 0;
        for (int nodeLevel = 0; nodeLevel <= maxLevelBvh; nodeLevel++)
        {
            int levelNodeCount = Mathf.CeilToInt(Mathf.Pow(2f, nodeLevel));
            numBvhNodes += levelNodeCount;
        }
        bvh.Capacity = numBvhNodes;
        for (int i = 0; i < numBvhNodes; i++)
        {
            bvh.Add(new BvhSphereNode());
        }

        List<int> cutValues = new(){ 1 << 30 };

        for (int numLevels = 0, bvhIndex = 0; numLevels < numLevelsBVHMorton; numLevels++)
        {
            for (int i = 0; i < Mathf.Pow(2, numLevels); i++, bvhIndex++)
            {
                int minValue = 0;
                if(i > 0) minValue = cutValues[i - 1];

                TrianglesSpan triSpan = GetTrianglesInMortonCodeSpan(triangles, minValue, cutValues[i]);
                bvh[bvhIndex].boundingSphere = BoundingSphere.Create(triSpan.triangles);
                bvh[bvhIndex].childrenORspan = new int[2] { -triSpan.firstElementIndex, triSpan.triangles.Count };
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

    private void SplitLeafNodesWithSah(int maxTrisPerBVHNode = maxTrisPerBvhNode)
    {
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
                    bvh[childIndex].boundingSphere = BoundingSphere.Create(childTris);
                    bvh[childIndex].childrenORspan = new int[2] { -curNode.FirstTriIndex(), childTris.Count };

                    childIndex++;
                    childTris = triangles.GetRange(partIndex, curNode.TrisCount() - childTris.Count);
                    bvh[childIndex].boundingSphere = BoundingSphere.Create(childTris);
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

    private void VisualizeAllBvhNodes()
    {
        int numNodes = bvh.Count;
        sphericalBvhNodes.Capacity = numNodes;

        for (int i = 0; i < numNodes; i++)
        {
            BvhSphereNode curNode = bvh[i];
            sphericalBvhNodes.Add(Instantiate(sphericalNodePrefab));
            sphericalBvhNodes.Last().transform.position = curNode.boundingSphere.center;
            sphericalBvhNodes.Last().transform.localScale = Vector3.one * (curNode.boundingSphere.radius * 2f);
        }
    }

    private void VisualizeBvhNodes()
    {
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
    }
}
