// Phase 1, step 5: run the glare pyramid on the frames exported by
// experiments/02_slang_plumbing/unity_parity.py and write every mip and upsample level back
// as raw float32 RGBA (row 0 = texel y = 0, no header) for the Python comparison. Weights with
// one level more than the pyramid (Phase 2) also run the full-resolution level (_unity_full).
// The per-level parameter count comes from glare_params.json, which unity_parity.py export writes
// from glare.slang's LevelMLP, so it can't drift from the network the HLSL was compiled from.
// Weights with the spike section (spikeParamCount > 0 in the layout; decision log 2026-11-04)
// also run the spike term after the full-resolution level: every ray's streak passes on mip 0,
// summed per family, then SpikeComposite (_unity_even, _unity_odd, _unity_spikes). The rays are
// streaked from mip 0 or from the input frame, as the layout's spikeSource says. Weights with the
// per-ray cross filter (crossParamCount > 0; decision log 2026-11-19) run every streak pass plain
// and RayCrossAccumulate adds each ray to its family. The pyramid's mip count is the layout's
// `mips` (unity_parity.py export --mips, the run's _mips<n>), so 1-mip models (2026-11-10) run too.
// Weights with per-ray brightness (brightParamCount > 0, after the cross filter) run
// RayBrightAccumulate in place of RayCrossAccumulate and SpikeCompositeBright in place of
// SpikeComposite.
// The dispatch sequence lives in Runtime/GlareChain.cs, shared with the URP renderer feature
// (Runtime/NeuralGlareFeature.cs), so parity checks the code that is timed and shipped.
using System;
using System.IO;
using NeuralGlare;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

static class ParityHarness
{
    const string ShaderDir = "Assets/NeuralGlare/Shaders/";
    const string WeightsPath = "Assets/NeuralGlare/Data/glare_params.bytes";
    const string LayoutPath = "Assets/NeuralGlare/Data/glare_params.json";   // unity_parity.py UNITY_LAYOUT

    [Serializable] class Frame { public string name; public int width; public int height; }
    [Serializable] class Manifest { public Frame[] frames; }
    [Serializable] class RunInfo
    {
        public string unityVersion, graphicsDeviceType, graphicsDeviceName, graphicsDeviceVersion;
    }

    // Assets/ -> NeuralGlareURP/ -> unity/ -> engines/ -> repo root
    static string ParityDir => Path.GetFullPath(Path.Combine(Application.dataPath,
        "../../../../outputs/02_slang_plumbing/unity_parity"));

    [MenuItem("NeuralGlare/Run Parity")]
    static void Run()
    {
        var manifest = JsonUtility.FromJson<Manifest>(
            File.ReadAllText(Path.Combine(ParityDir, "frames.json")));
        var shaders = new GlareShaders
        {
            prefilter = LoadShader("GlarePrefilter"),
            blurH = LoadShader("GlareBlurH"),
            blurV = LoadShader("GlareBlurV"),
            upsample = LoadShader("GlareUpsample"),
            streak = LoadShader("GlareStreak"),
            streakAccumulate = LoadShader("GlareStreakAccumulate"),
            rayCrossAccumulate = LoadShader("GlareRayCrossAccumulate"),
            rayBrightAccumulate = LoadShader("GlareRayBrightAccumulate"),
            spikeComposite = LoadShader("GlareSpikeComposite"),
            spikeCompositeBright = LoadShader("GlareSpikeCompositeBright"),
        };
        GlareLayout layout = GlareLayout.Parse(LoadText(LayoutPath).text, LayoutPath);
        float[] weights = layout.ReadWeights(LoadText(WeightsPath).bytes, WeightsPath);

        using (var chain = new GlareChain(shaders, layout, weights))
            foreach (Frame frame in manifest.frames)
                RunFrame(frame, chain);

        var info = new RunInfo
        {
            unityVersion = Application.unityVersion,
            graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
            graphicsDeviceName = SystemInfo.graphicsDeviceName,
            graphicsDeviceVersion = SystemInfo.graphicsDeviceVersion,
        };
        File.WriteAllText(Path.Combine(ParityDir, "unity_run.json"), JsonUtility.ToJson(info, true));
        Debug.Log($"Parity: {manifest.frames.Length} frames on {info.graphicsDeviceVersion} -> {ParityDir}");
    }

    static void RunFrame(Frame frame, GlareChain chain)
    {
        var input = new Texture2D(frame.width, frame.height, TextureFormat.RGBAFloat,
                                  mipChain: false, linear: true);
        try
        {
            input.SetPixelData(ReadFloats(Path.Combine(ParityDir, frame.name + "_input.f32"),
                                          frame.width * frame.height * 4), 0);
            input.Apply(updateMipmaps: false);

            chain.Resize(frame.width, frame.height);
            using (var cmd = new CommandBuffer { name = "NeuralGlare parity" })
            {
                chain.Record(cmd, input);
                Graphics.ExecuteCommandBuffer(cmd);
            }

            for (int i = 0; i < chain.Mips.Length; ++i)
                WriteFloats(chain.Mips[i], Path.Combine(ParityDir, $"{frame.name}_unity_mip{i}.f32"));
            for (int k = 0; k < chain.Ups.Length; ++k)
                WriteFloats(chain.Ups[k], Path.Combine(ParityDir, $"{frame.name}_unity_up{k}.f32"));
            if (chain.HasFullLevel)
                WriteFloats(chain.Full, Path.Combine(ParityDir, $"{frame.name}_unity_full.f32"));
            if (chain.HasSpikes)
            {
                WriteFloats(chain.FamilyEven, Path.Combine(ParityDir, $"{frame.name}_unity_even.f32"));
                WriteFloats(chain.FamilyOdd, Path.Combine(ParityDir, $"{frame.name}_unity_odd.f32"));
                WriteFloats(chain.Output, Path.Combine(ParityDir, $"{frame.name}_unity_spikes.f32"));
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(input);
        }
    }

    static ComputeShader LoadShader(string name)
    {
        var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ShaderDir + name + ".compute");
        return shader != null ? shader : throw new FileNotFoundException(ShaderDir + name + ".compute");
    }

    static TextAsset LoadText(string path)
    {
        var asset = AssetDatabase.LoadAssetAtPath<TextAsset>(path);
        return asset != null ? asset : throw new FileNotFoundException($"{path}: run unity_parity.py export");
    }

    static float[] ReadFloats(string path, int count)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * sizeof(float))
            throw new InvalidDataException($"{path}: {bytes.Length} bytes, expected {count * sizeof(float)}");
        var floats = new float[count];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    static void WriteFloats(RenderTexture rt, string path)
    {
        var request = AsyncGPUReadback.Request(rt);
        request.WaitForCompletion();
        if (request.hasError)
            throw new IOException($"GPU readback failed for {path}");
        NativeArray<byte> bytes = request.GetData<float>().Reinterpret<byte>(sizeof(float));
        File.WriteAllBytes(path, bytes.ToArray());
    }
}
