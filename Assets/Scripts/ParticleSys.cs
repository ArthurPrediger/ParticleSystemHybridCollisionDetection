using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static Unity.IO.LowLevel.Unsafe.AsyncReadManagerMetrics;

public struct Particle
{
    public Particle(Vector3 position)
    {
        pos = position;
    }

    public Vector3 pos;
}

public class ParticleSys : MonoBehaviour
{
    [SerializeField]
    private ComputeShader csUpdate;

    private Material partSysMat;

    private Mesh mesh;

    private RenderTexture renderTexture;

    private Particle[] particles;

    private List<Vector3> particlesPos = new List<Vector3>();

    private HashSet<Vector3> vertices = new HashSet<Vector3>();

    private ComputeBuffer particlesCB;

    private float aliveTime = 0.0f;

    // Start is called before the first frame update
    void Start()
    {
        mesh = GetComponent<MeshFilter>().mesh;

        foreach (var pos in mesh.vertices)
        {
            if(vertices.Add(pos))
                particlesPos.Add(pos);
        }

        List<int> indices = new List<int>();
        foreach (var i in mesh.GetIndices(0))
        {
            indices.Add(indices.Count % particlesPos.Count);
        }

        Mesh partMesh = new Mesh();
        partMesh.name = "Particle Mesh";
        partMesh.SetVertices(particlesPos);
        partMesh.SetTriangles(indices, 0);
        GetComponent<MeshFilter>().mesh = partMesh;

        particlesCB = new ComputeBuffer(particlesPos.Count, sizeof(float) * 3);
        particlesCB.SetData(particlesPos.ToArray());

        partSysMat = GetComponent<MeshRenderer>().material;
        partSysMat.SetBuffer("particlesPos", particlesCB);

        csUpdate.SetBuffer(0, "particlesPos", particlesCB);
    }

    // Update is called once per frame
    void Update()
    {
        aliveTime += Time.deltaTime;
        csUpdate.SetFloat(Shader.PropertyToID("deltaTime"), Time.deltaTime);
        csUpdate.SetFloat(Shader.PropertyToID("aliveTime"), aliveTime);
        csUpdate.Dispatch(csUpdate.FindKernel("PSUpdate"), particlesPos.Count, 1, 1);

        if (aliveTime > 2.0f) aliveTime = 0.0f;
    }

    void OnDestroy()
    {
        if (particlesCB != null)
        {
            particlesCB.Release();
        }
    }

    void OnRenderObject()
    {
    }
}
