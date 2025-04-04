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

        particleSys = GetComponent<ParticleSys>();

        SetNumParticlesBenchmark(0f);
    }

    // Update is called once per frame
    void Update()
    {
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
                }
                else
                {
                    curCollisionDetectionMethod = 0;
                    isBenchmarkRunning = false;
                    particleSys.enabled = false;
                    runBenchButton.gameObject.SetActive(true);
                    quitButton.gameObject.SetActive(true);
                    numParticlesBenchText.enabled = true;
                    numParticlesBenchScroll.gameObject.SetActive(true);

                    ComputePresentResults();
                }
            }

            cameras[indexCurActiveCamera].SetActive(true);
            cameraActiveTimeSteps = 0;

            isBenchmarkRunning = false;
            particleSys.enabled = false;
            StartCoroutine(WaitCameraChange());
        }
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

        particleSys.ResetBenchmarks();
        StartCoroutine(RunParticleSystemSetup());

        collisionDetectionMethods = new() {
            "SetScreenSpaceCollisionActive",
            "SetVolumeStructureCollisionActive",
            "SetHybridCollisionActive",
        };

        GetComponent<ParticleSys>().Invoke(collisionDetectionMethods[curCollisionDetectionMethod], 0f);
    }

    private IEnumerator RunParticleSystemSetup()
    {
        yield return null;

        int yLayers = (int)Mathf.Pow(2f, curScrollbarStep);
        particleSys.SetupParticleSystemData(yLayers);
        particleSys.enabled = true;
        loadingBenchText.enabled = false;
        isBenchmarkRunning = true;
    }

    public void QuitBenchmark()
    {
        Application.Quit();
    }

    void ComputePresentResults()
    {
        resultsBenchText.text = "<align=center>Averages of Simulation Steps Results:</align>\n\n";

        var benchmarkTimings = particleSys.GetBenchmarkTimings();

        string filePath = Application.streamingAssetsPath + "/results.csv";
        StreamWriter writer = new StreamWriter(filePath);

        foreach (var benchTiming in benchmarkTimings)
        {
            writer.WriteLine($"{benchTiming.Item1};ms");

            benchTiming.Item2.RemoveAt(0);
            float runningAverage = 0;
            for (int i = 0; i < benchTiming.Item2.Count; i++)
            {
                runningAverage = (runningAverage * (float)i + benchTiming.Item2[i]) / (float)(i + 1);

                writer.WriteLine($"{i};{benchTiming.Item2[i]}");
            }

            writer.WriteLine($"Average;{runningAverage}");

            resultsBenchText.text += benchTiming.Item1 + ": " + runningAverage.ToString("F4") + "ms\n";
        }

        writer.Close();
        resultsBenchText.enabled = true;
    }

    public void SetNumParticlesBenchmark(Single single)
    {
        curScrollbarStep = Mathf.FloorToInt(Mathf.Max((single - 0.0001f), 0f) * (float)numParticlesBenchScroll.numberOfSteps);

        numParticlesBenchText.text = "Number of Particles " + (particleSys.numParticlesXZ * particleSys.numParticlesXZ * (int)Mathf.Pow(2f, curScrollbarStep)).ToString();
    }
}
