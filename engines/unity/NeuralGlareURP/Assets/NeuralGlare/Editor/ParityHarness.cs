// Phase 1, step 5: run the glare pyramid on the frames exported by
// experiments/02_slang_plumbing/unity_parity.py and write every mip and upsample level back
// as raw float32 RGBA (row 0 = texel y = 0, no header) for the Python comparison. Weights with
// one level more than the pyramid (Phase 2) also run the full-resolution level (_unity_full).
// The per-level parameter count comes from glare_params.json, which unity_parity.py export writes
// from glare.slang's LevelMLP, so it can't drift from the network the HLSL was compiled from.
// Weights with the spike section (spikeParamCount > 0 in the layout; decision log 2026-11-04)
// also run the spike term after the full-resolution level: every ray's streak passes on mip 0,
// summed per family, then SpikeComposite (_unity_even, _unity_odd, _unity_spikes). The rays are
// streaked from mip 0 or from the input frame, as the layout's spikeSource says.
using System;
using System.IO;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

static class ParityHarness
{
    const string ShaderDir = "Assets/NeuralGlare/Shaders/";
    const string WeightsPath = "Assets/NeuralGlare/Data/glare_params.bytes";
    const string LayoutPath = "Assets/NeuralGlare/Data/glare_params.json";   // unity_parity.py UNITY_LAYOUT
    const int MaxIterations = 6;    // URPBloomParams.max_iterations in train_pyramid.py

    [Serializable] class Frame { public string name; public int width; public int height; }
    [Serializable] class Manifest { public Frame[] frames; }
    [Serializable] class Layout
    {
        public int inputs, hidden, outputs, paramCount, levels;
        public int spikeParamCount, profileCount, rays, passes;     // 0 without the spike term
        public string spikeSource;                                  // "mip0" or "input"
        public int SpikeFloats => spikeParamCount > 0 ? spikeParamCount + profileCount : 0;
    }
    class Shaders
    {
        public ComputeShader prefilter, blurH, blurV, upsample, streak, streakAccumulate, spikeComposite;
    }
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
        var shaders = new Shaders
        {
            prefilter = LoadShader("GlarePrefilter"),
            blurH = LoadShader("GlareBlurH"),
            blurV = LoadShader("GlareBlurV"),
            upsample = LoadShader("GlareUpsample"),
            streak = LoadShader("GlareStreak"),
            streakAccumulate = LoadShader("GlareStreakAccumulate"),
            spikeComposite = LoadShader("GlareSpikeComposite"),
        };

        Layout layout = LoadLayout();
        float[] weights = LoadWeights(layout);
        using var paramBuffer = new ComputeBuffer(weights.Length, sizeof(float));
        paramBuffer.SetData(weights);

        foreach (Frame frame in manifest.frames)
            RunFrame(frame, shaders, paramBuffer, layout);

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

    static void RunFrame(Frame frame, Shaders shaders, ComputeBuffer paramBuffer, Layout layout)
    {
        ComputeShader prefilter = shaders.prefilter, blurH = shaders.blurH, blurV = shaders.blurV,
                      upsample = shaders.upsample;
        int levels = layout.levels;
        var input = new Texture2D(frame.width, frame.height, TextureFormat.RGBAFloat,
                                  mipChain: false, linear: true);
        input.SetPixelData(ReadFloats(Path.Combine(ParityDir, frame.name + "_input.f32"),
                                      frame.width * frame.height * 4), 0);
        input.Apply(updateMipmaps: false);

        Vector2Int[] sizes = MipSizes(frame.width, frame.height);
        int n = sizes.Length - 1;
        // Phase 1 weights: one network per pyramid level. Phase 2 weights: one more, for the
        // full-resolution level whose "mip" is the input frame (train_points.py glare()).
        bool fullLevel = levels == n + 1;
        if (levels != n && !fullLevel)
            throw new InvalidDataException($"{frame.name}: {n} upsample levels, weights have {levels}");
        if (layout.spikeParamCount > 0 && !fullLevel)
            throw new InvalidDataException($"{frame.name}: spike weights without the full-resolution level");

        var mips = new RenderTexture[n + 1];
        var ups = new RenderTexture[n];        // ups[n] is mips[n]
        RenderTexture full = null;
        RenderTexture[] families = null;   // even, odd
        RenderTexture spikes = null;
        try
        {
            // Downsample: mip 0 = prefilter, mip i = BlurV(BlurH(mip i-1)) at half the size
            mips[0] = Target(sizes[0]);
            Dispatch(prefilter, input, mips[0]);
            for (int i = 1; i <= n; ++i)
            {
                RenderTexture blurred = Target(sizes[i]);
                Dispatch(blurH, mips[i - 1], blurred);
                mips[i] = Target(sizes[i]);
                Dispatch(blurV, blurred, mips[i]);
                blurred.Release();
            }

            // Upsample: level k reads mip k and the output of level k + 1
            upsample.SetBuffer(0, "_GlareParams", paramBuffer);
            for (int k = n - 1; k >= 0; --k)
            {
                ups[k] = Target(sizes[k]);
                upsample.SetTexture(0, "_SourceLow", k + 1 < n ? ups[k + 1] : mips[n]);
                upsample.SetInt("_ParamOffset", k * layout.paramCount);
                Dispatch(upsample, mips[k], ups[k]);
            }

            // Full-resolution level: the same kernel, reading the input frame and up[0]
            if (fullLevel)
            {
                full = Target(new Vector2Int(frame.width, frame.height));
                upsample.SetTexture(0, "_SourceLow", ups[0]);
                upsample.SetInt("_ParamOffset", n * layout.paramCount);
                Dispatch(upsample, input, full);
            }

            // Spike term: every ray streaked from mip 0, summed per family, mixed into the frame
            if (layout.spikeParamCount > 0)
            {
                bool fromInput = layout.spikeSource == "input";
                families = RunRays(shaders, paramBuffer, layout, fromInput ? input : mips[0],
                                   fromInput ? new Vector2Int(frame.width, frame.height) : sizes[0]);
                spikes = Target(new Vector2Int(frame.width, frame.height));
                ComputeShader composite = shaders.spikeComposite;
                composite.SetBuffer(0, "_GlareParams", paramBuffer);
                composite.SetTexture(0, "_SpikeEven", families[0]);
                composite.SetTexture(0, "_SpikeOdd", families[1]);
                composite.SetInt("_ParamOffset", levels * layout.paramCount);
                Dispatch(composite, full, spikes);
            }

            for (int i = 0; i <= n; ++i)
                WriteFloats(mips[i], Path.Combine(ParityDir, $"{frame.name}_unity_mip{i}.f32"));
            for (int k = 0; k < n; ++k)
                WriteFloats(ups[k], Path.Combine(ParityDir, $"{frame.name}_unity_up{k}.f32"));
            if (fullLevel)
                WriteFloats(full, Path.Combine(ParityDir, $"{frame.name}_unity_full.f32"));
            if (spikes != null)
            {
                WriteFloats(families[0], Path.Combine(ParityDir, $"{frame.name}_unity_even.f32"));
                WriteFloats(families[1], Path.Combine(ParityDir, $"{frame.name}_unity_odd.f32"));
                WriteFloats(spikes, Path.Combine(ParityDir, $"{frame.name}_unity_spikes.f32"));
            }
        }
        finally
        {
            if (families != null) foreach (RenderTexture rt in families) if (rt != null) rt.Release();
            if (spikes != null) spikes.Release();
            foreach (RenderTexture rt in mips) if (rt != null) rt.Release();
            foreach (RenderTexture rt in ups) if (rt != null) rt.Release();
            if (full != null) full.Release();
            UnityEngine.Object.DestroyImmediate(input);
        }
    }

    // Each ray: passes - 1 Streak passes from source (mip 0 or the input frame), then
    // StreakAccumulate onto its family's sum (ray k is in family k % 2), as train_points.py
    // spikes(). Returns the two sums.
    static RenderTexture[] RunRays(Shaders shaders, ComputeBuffer paramBuffer, Layout layout,
                                   Texture source, Vector2Int size)
    {
        int profile = layout.levels * layout.paramCount + layout.spikeParamCount;
        var families = new[] { Cleared(size), Cleared(size) };
        foreach (ComputeShader shader in new[] { shaders.streak, shaders.streakAccumulate })
        {
            shader.SetBuffer(0, "_GlareParams", paramBuffer);
            shader.SetInt("_ParamOffset", profile);
        }
        for (int ray = 0; ray < layout.rays; ++ray)
        {
            Texture s = source;
            for (int stage = 0; stage < layout.passes; ++stage)
            {
                bool last = stage == layout.passes - 1;
                ComputeShader shader = last ? shaders.streakAccumulate : shaders.streak;
                shader.SetInt("_Ray", ray);
                shader.SetInt("_Stage", stage);
                if (last)
                    shader.SetTexture(0, "_Accum", families[ray % 2]);
                RenderTexture next = Target(size);
                Dispatch(shader, s, next);
                if (s != source)
                    ((RenderTexture)s).Release();
                if (last)
                {
                    families[ray % 2].Release();
                    families[ray % 2] = next;
                }
                else
                    s = next;
            }
        }
        return families;
    }

    static RenderTexture Cleared(Vector2Int size)
    {
        RenderTexture rt = Target(size);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;
        GL.Clear(false, true, Color.clear);
        RenderTexture.active = previous;
        return rt;
    }

    // As mip_sizes() in pyramid_bloom.py (URP Bloom, downscale "half"). Integer log2, so
    // powers of two can't round down the way Mathf.Log(x, 2) can.
    static Vector2Int[] MipSizes(int width, int height)
    {
        int w = width >> 1, h = height >> 1;
        int count = Mathf.Clamp(FloorLog2(Mathf.Max(w, h)) - 1, 1, MaxIterations);
        var sizes = new Vector2Int[count];
        sizes[0] = new Vector2Int(w, h);
        for (int i = 1; i < count; ++i)
        {
            w = Mathf.Max(1, w >> 1);
            h = Mathf.Max(1, h >> 1);
            sizes[i] = new Vector2Int(w, h);
        }
        return sizes;
    }

    static int FloorLog2(int n)
    {
        int log = -1;
        for (; n > 0; n >>= 1)
            ++log;
        return log;
    }

    static RenderTexture Target(Vector2Int size)
    {
        var rt = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.ARGBFloat,
                                   RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true,
            useMipMap = false,
            filterMode = FilterMode.Point,
        };
        rt.Create();
        return rt;
    }

    // Each .compute holds one kernel, so its index is 0. Target size comes from _Target itself.
    static void Dispatch(ComputeShader shader, Texture source, RenderTexture target)
    {
        shader.SetTexture(0, "_Source", source);
        shader.SetTexture(0, "_Target", target);
        shader.Dispatch(0, (target.width + 7) / 8, (target.height + 7) / 8, 1);
    }

    static ComputeShader LoadShader(string name)
    {
        var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ShaderDir + name + ".compute");
        return shader != null ? shader : throw new FileNotFoundException(ShaderDir + name + ".compute");
    }

    static Layout LoadLayout()
    {
        var asset = AssetDatabase.LoadAssetAtPath<TextAsset>(LayoutPath);
        if (asset == null)
            throw new FileNotFoundException($"{LayoutPath}: run unity_parity.py export");
        var layout = JsonUtility.FromJson<Layout>(asset.text);
        if (layout.paramCount <= 0 || layout.levels <= 0)
            throw new InvalidDataException($"{LayoutPath}: paramCount {layout.paramCount}, levels {layout.levels}");
        if (layout.spikeParamCount > 0 && (layout.profileCount <= 0 || layout.rays <= 0 || layout.passes <= 0))
            throw new InvalidDataException($"{LayoutPath}: spike section without profileCount / rays / passes");
        if (layout.spikeParamCount > 0 && layout.spikeSource != "mip0" && layout.spikeSource != "input")
            throw new InvalidDataException($"{LayoutPath}: spikeSource \"{layout.spikeSource}\", expected mip0 or input");
        return layout;
    }

    static float[] LoadWeights(Layout layout)
    {
        byte[] bytes = AssetDatabase.LoadAssetAtPath<TextAsset>(WeightsPath).bytes;
        if (bytes.Length != (layout.levels * layout.paramCount + layout.SpikeFloats) * sizeof(float))
            throw new InvalidDataException($"{WeightsPath}: {bytes.Length} bytes, {LayoutPath} says {layout.levels} levels of {layout.paramCount} floats + {layout.SpikeFloats} spike floats");
        var weights = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, weights, 0, bytes.Length);
        return weights;
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
