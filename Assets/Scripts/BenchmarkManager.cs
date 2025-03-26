using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class BenchmarkManager : MonoBehaviour
{
    [SerializeField]
    private List<GameObject> cameras = new();

    private int indexCurActiveCamera = 0;
    private float cameraActiveTime = 0.0f;
    private float cameraActiveLifetime;

    private List<string> collisionDetectionMethods;
    private int curCollisionDetectionMethod = 0;   

    // Start is called before the first frame update
    void Start()
    {
        foreach (GameObject obj in cameras)
        {
            obj.GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
            obj.SetActive(false);
        }

        cameras.First().SetActive(true);

        cameraActiveLifetime = GetComponent<ParticleSys>().particlesLifetime;

        collisionDetectionMethods = new() {
            "SetHybridCollisionActive",
            "SetScreenSpaceCollisionActive",
            "SetVolumeStructureCollisionActive",
        };

        GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
    }

    // Update is called once per frame
    void Update()
    {
        cameraActiveTime += Time.deltaTime;

        if(cameraActiveTime > cameraActiveLifetime)
        {
            cameras[indexCurActiveCamera++].SetActive(false);

            if (indexCurActiveCamera >= cameras.Count)
            {
                indexCurActiveCamera = 0;
                curCollisionDetectionMethod = (curCollisionDetectionMethod + 1) % collisionDetectionMethods.Count;
                GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
            }

            cameras[indexCurActiveCamera].SetActive(true);
            cameraActiveTime = 0f;
        }
    }
}
