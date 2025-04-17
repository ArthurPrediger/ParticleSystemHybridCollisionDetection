#define PERFORMANCE_BENCHMARK
//#define ACCURACY_BENCHMARK

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class BenchmarkManager : MonoBehaviour
{
    private ParticleSys particleSys;

    [SerializeField]
    private List<GameObject> cameras = new();

    private int indexCurActiveCamera = 0;
    private int cameraActiveTimeSteps = 0;
    private float cameraActiveLifetimeSteps;

    private List<string> collisionDetectionMethods;
    private List<string> collisionDetectionMethodsNames;
    private int curCollisionDetectionMethod = 0;
    
    private bool isBenchmarkRunning = false;

    [SerializeField]
    private Button runBenchButton;
    [SerializeField]
    private Button quitButton;
    [SerializeField]
    private TextMeshProUGUI loadingBenchText;
    [SerializeField]
    private TextMeshProUGUI resultsBenchText;
    [SerializeField]
    private TextMeshProUGUI numParticlesBenchText;
    [SerializeField]
    private Scrollbar numParticlesBenchScroll;
    [SerializeField]
    private TextMeshProUGUI frameTimeText;
    [SerializeField]
    private TextMeshProUGUI activeColDetecMethodText;

    int curScrollbarStep = 0;


    // Start is called before the first frame update
    void Start()
    {
        foreach (GameObject obj in cameras)
        {
            obj.GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
            obj.SetActive(false);
        }

        cameras.First().SetActive(true);

        cameraActiveLifetimeSteps = GetComponent<ParticleSys>().particlesLifetimeSteps;

        loadingBenchText.enabled = false;
        resultsBenchText.enabled = false;
        activeColDetecMethodText.enabled = false;
        frameTimeText.enabled = true;

        particleSys = GetComponent<ParticleSys>();

        SetNumParticlesBenchmark(0f);
    }

    // Update is called once per frame
    void Update()
    {
        frameTimeText.text = (1f / Time.deltaTime).ToString("F1") + " FPS (" + (Time.deltaTime * 1000f).ToString("F1") + "ms)";

        if (!isBenchmarkRunning) return;

        if(cameraActiveTimeSteps++ >= cameraActiveLifetimeSteps)
        {
            cameras[indexCurActiveCamera++].SetActive(false);

            if (indexCurActiveCamera >= cameras.Count)
            {
                indexCurActiveCamera = 0;
                curCollisionDetectionMethod++;
                if (curCollisionDetectionMethod < collisionDetectionMethods.Count)
                {
                    particleSys.Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
                    activeColDetecMethodText.text = collisionDetectionMethodsNames[curCollisionDetectionMethod];
                }
                else
                {
                    curCollisionDetectionMethod = 0;
                    isBenchmarkRunning = false;
                    particleSys.enabled = false;
                    activeColDetecMethodText.enabled = false;
                    runBenchButton.gameObject.SetActive(true);
                    quitButton.gameObject.SetActive(true);
                    numParticlesBenchText.enabled = true;
                    numParticlesBenchScroll.gameObject.SetActive(true);

#if PERFORMANCE_BENCHMARK
                    ComputePresentPerformanceResults();
#elif ACCURACY_BENCHMARK
                    ComputePresentAccuracyResults();
#endif
                }
            }

            cameras[indexCurActiveCamera].SetActive(true);
            cameraActiveTimeSteps = 0;

            isBenchmarkRunning = false;
            particleSys.enabled = false;
            StartCoroutine(WaitCameraChange());
        }
    }

    public int GetCamerasCount()
    {
        return cameras.Count;
    }

    IEnumerator WaitCameraChange()
    {
        yield return new WaitForSeconds(1f);

        if (!resultsBenchText.enabled)
        {
            isBenchmarkRunning = true;
            particleSys.enabled = true;
        }
    }

    public void StartBenchmark()
    {
        loadingBenchText.enabled = true;
        resultsBenchText.enabled = false;
        numParticlesBenchText.enabled = false;
        runBenchButton.gameObject.SetActive(false);
        quitButton.gameObject.SetActive(false);
        numParticlesBenchScroll.gameObject.SetActive(false);
        Canvas.ForceUpdateCanvases();

#if PERFORMANCE_BENCHMARK
        particleSys.ResetBenchmarkTimings();
#elif ACCURACY_BENCHMARK
        particleSys.ResetBenchmarkCollisons();
#endif
        StartCoroutine(RunParticleSystemSetup());

        collisionDetectionMethods = new() {
            "SetScreenSpaceCollisionActive",
            "SetVolumeStructureCollisionActive",
            "SetHybridCollisionActive",
        };

        collisionDetectionMethodsNames = particleSys.GetCollisionDetectionMethodsNames();

        GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
        activeColDetecMethodText.text = collisionDetectionMethodsNames[curCollisionDetectionMethod];
    }

    private IEnumerator RunParticleSystemSetup()
    {
        yield return null;

        int yLayers = (int)Mathf.Pow(2f, curScrollbarStep);
        particleSys.SetupParticleSystemData(yLayers);
        particleSys.enabled = true;
        loadingBenchText.enabled = false;
        activeColDetecMethodText.enabled = true;
        isBenchmarkRunning = true;
    }

    public void QuitBenchmark()
    {
        Application.Quit();
    }

#if PERFORMANCE_BENCHMARK
    void ComputePresentPerformanceResults()
    {
        resultsBenchText.text = "<align=center>Averages of Simulation Steps Results:</align>\n\n";

        var benchmarkTimings = particleSys.GetBenchmarkTimings();

        string filePath = Application.streamingAssetsPath + "/results.csv";
        StreamWriter writer = new StreamWriter(filePath);

        int j = 0;
        foreach (var benchTiming in benchmarkTimings)
        {
            writer.WriteLine($"{collisionDetectionMethodsNames[j]};ms");

            for(int i = cameras.Count - 1; i >= 0; i--)
            {
                benchTiming.RemoveAt(particleSys.particlesLifetimeSteps * i);
            }
            float runningAverage = 0;
            for (int i = 0; i < benchTiming.Count; i++)
            {
                runningAverage = (runningAverage * (float)i + benchTiming[i]) / (float)(i + 1);

                writer.WriteLine($"{i};{benchTiming[i]}");
            }

            writer.WriteLine($"Average;{runningAverage}");

            resultsBenchText.text += collisionDetectionMethodsNames[j] + ": " + runningAverage.ToString("F4") + "ms\n";
            j++;
        }

        writer.Close();
        resultsBenchText.enabled = true;
    }
#endif

#if ACCURACY_BENCHMARK
    void ComputePresentAccuracyResults()
    {
        resultsBenchText.text = "<align=center>Total Collisions of the Simulation Results:</align>\n\n";

        var benchmarkCollisons = particleSys.GetBenchmarkCollisions();

        string filePath = Application.streamingAssetsPath + "/results.csv";
        StreamWriter writer = new StreamWriter(filePath);

        int j = 0;
        foreach (var benchCollisons in benchmarkCollisons)
        {
            writer.WriteLine($"{collisionDetectionMethodsNames[j]};collisions");

            int totalCollisons = 0;
            for (int i = 0; i < benchCollisons.Length; i++)
            {
                totalCollisons += benchCollisons[i];

                writer.WriteLine($"{i};{benchCollisons[i]}");
            }

            writer.WriteLine($"Total collisions;{totalCollisons}");

            resultsBenchText.text += collisionDetectionMethodsNames[j] + ": " + totalCollisons.ToString() + " collisions\n";
            j++;
        }

        writer.Close();
        resultsBenchText.enabled = true;
    }
#endif

    public void SetNumParticlesBenchmark(Single single)
    {
        curScrollbarStep = Mathf.FloorToInt(Mathf.Max((single - 0.0001f), 0f) * (float)numParticlesBenchScroll.numberOfSteps);

        numParticlesBenchText.text = "Number of Particles " + (particleSys.numParticlesXZ * particleSys.numParticlesXZ * (int)Mathf.Pow(2f, curScrollbarStep)).ToString();
    }
}
