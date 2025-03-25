using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class BenchmarkManager : MonoBehaviour
{
    [SerializeField]
    private List<Camera> cameras = new();

    private int indexCurActiveCamera = 0;
    private float cameraActiveTime = 0.0f;
    private float cameraActiveLifetime;

    private List<string> collisionDetectionMethods;
    private int curCollisionDetectionMethod = 0;   

    // Start is called before the first frame update
    void Start()
    {
        foreach (Camera cam in cameras)
        {
            cam.depthTextureMode = DepthTextureMode.Depth;
            cam.enabled = false;
        }

        cameras.First().enabled = true;

        cameraActiveLifetime = GetComponent<ParticleSys>().particlesLifetime;

        collisionDetectionMethods = new() {
            "SetScreenSpaceCollisionActive",
            "SetVolumeStructureCollisionActive",
            "SetHybridCollisionActive"
        };

        GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
    }

    // Update is called once per frame
    void Update()
    {
        cameraActiveTime += Time.deltaTime;

        if(cameraActiveTime > cameraActiveLifetime)
        {
            cameras[indexCurActiveCamera++].enabled = false;
            
            if(indexCurActiveCamera >= cameras.Count)
            {
                indexCurActiveCamera = 0;
                curCollisionDetectionMethod = (curCollisionDetectionMethod + 1) % collisionDetectionMethods.Count;
                GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
            }

            cameras[indexCurActiveCamera].enabled = true;
            cameraActiveTime = 0f;
        }
    }
}
