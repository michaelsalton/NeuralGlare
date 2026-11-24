// GPU timing of the NeuralGlare passes, by the 2026-10-29 protocol (decision log): warm-up, then
// `windows` windows; per marker, each window's median frame time, reported as the median and range
// over windows (Unreal: median of 5 ProfileGPU captures). Put it on any GameObject in the timing
// scene. Command line: -ngTimingOut <path.json> writes there and quits when done; without it the
// result goes to Application.persistentDataPath and the player keeps running.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace NeuralGlare
{
    public sealed class GlareTimer : MonoBehaviour
    {
        [SerializeField] float warmupSeconds = 10f;
        [SerializeField] float windowSeconds = 2f;
        [SerializeField] int windows = 5;

        static readonly ProfilingSampler[] Samplers =
            { GlareChain.Total, GlareChain.Pyramid, GlareChain.Levels, GlareChain.Rays, GlareChain.Composite };

        [Serializable] class MarkerResult
        {
            public string name;
            public float medianMs, minMs, maxMs;   // over the windows' medians
            public float[] windowMedianMs;
            public int frames;
        }

        [Serializable] class TimingResult
        {
            public string unityVersion, graphicsDeviceType, graphicsDeviceName, graphicsDeviceVersion, date;
            public string weights;     // the weight file loaded (NeuralGlareFeature.WeightsSource)
            public bool developmentBuild;
            public int screenWidth, screenHeight, glareWidth, glareHeight;
            public float warmupSeconds, windowSeconds;
            public MarkerResult[] markers;
        }

        List<float>[][] samples;    // [window][marker], ms per frame
        float start;
        string outPath;
        bool quitWhenDone;

        void Start()
        {
            QualitySettings.vSyncCount = 0;           // protocol: vsync and frame cap off
            Application.targetFrameRate = -1;
            outPath = Arg("-ngTimingOut");
            quitWhenDone = outPath != null;
            outPath ??= Path.Combine(Application.persistentDataPath, "neuralglare_timing.json");
            if (!SystemInfo.supportsGpuRecorder)
                Debug.LogError("GlareTimer: this device/API has no GPU recorder; no GPU times");
            foreach (ProfilingSampler s in Samplers)
                s.enableRecording = true;
            samples = Enumerable.Range(0, windows)
                .Select(_ => Samplers.Select(_ => new List<float>()).ToArray()).ToArray();
            start = Time.realtimeSinceStartup;
        }

        void Update()
        {
            float t = Time.realtimeSinceStartup - start - warmupSeconds;
            if (t < 0)
                return;
            int w = (int)(t / windowSeconds);
            if (w >= windows)
            {
                Write();
                enabled = false;
                if (quitWhenDone)
                    Application.Quit();
                return;
            }
            for (int m = 0; m < Samplers.Length; ++m)   // the samplers report the previous frame
                if (Samplers[m].gpuSampleCount > 0)
                    samples[w][m].Add(Samplers[m].gpuElapsedTime);
        }

        void Write()
        {
            var markers = new MarkerResult[Samplers.Length];
            for (int m = 0; m < Samplers.Length; ++m)
            {
                float[] medians = samples.Select(win => Median(win[m])).ToArray();
                markers[m] = new MarkerResult
                {
                    name = Samplers[m].name,
                    windowMedianMs = medians,
                    medianMs = Median(medians.ToList()),
                    minMs = medians.Min(),
                    maxMs = medians.Max(),
                    frames = samples.Sum(win => win[m].Count),
                };
                if (markers[m].frames == 0)
                    Debug.LogError($"GlareTimer: no GPU samples for {Samplers[m].name} (a Development Build? the feature enabled?)");
            }
            var result = new TimingResult
            {
                unityVersion = Application.unityVersion,
                graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
                graphicsDeviceName = SystemInfo.graphicsDeviceName,
                graphicsDeviceVersion = SystemInfo.graphicsDeviceVersion,
                date = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                weights = NeuralGlareFeature.WeightsSource,
                developmentBuild = Debug.isDebugBuild,
                screenWidth = Screen.width,
                screenHeight = Screen.height,
                glareWidth = GlareChain.LastSize.x,
                glareHeight = GlareChain.LastSize.y,
                warmupSeconds = warmupSeconds,
                windowSeconds = windowSeconds,
                markers = markers,
            };
            string dir = Path.GetDirectoryName(outPath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            File.WriteAllText(outPath, JsonUtility.ToJson(result, true));
            Debug.Log($"GlareTimer: {markers[0].name} {markers[0].medianMs:F3} ms ({markers[0].minMs:F3}-{markers[0].maxMs:F3}) -> {outPath}");
        }

        static float Median(List<float> values)
        {
            if (values.Count == 0)
                return float.NaN;
            List<float> sorted = values.OrderBy(v => v).ToList();
            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 1 ? sorted[mid] : 0.5f * (sorted[mid - 1] + sorted[mid]);
        }

        internal static string Arg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(args, name);
            return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        }
    }
}
