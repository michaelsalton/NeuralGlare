// Phase 1, step 5: run the glare pyramid on the frames exported by
// experiments/02_slang_plumbing/unity_parity.py and write every mip and upsample level back
// as raw float32 RGBA (row 0 = texel y = 0, no header) for the Python comparison.
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
    const int ParamCount = 82;      // LevelMLP.PARAM_COUNT = (2*16 + 16) + (16*2 + 2)
    const int MaxIterations = 6;    // URPBloomParams.max_iterations in train_pyramid.py

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
        ComputeShader prefilter = LoadShader("GlarePrefilter");
        ComputeShader blurH = LoadShader("GlareBlurH");
        ComputeShader blurV = LoadShader("GlareBlurV");
        ComputeShader upsample = LoadShader("GlareUpsample");

        float[] weights = LoadWeights();
        using var paramBuffer = new ComputeBuffer(weights.Length, sizeof(float));
        paramBuffer.SetData(weights);
        int levels = weights.Length / ParamCount;

        foreach (Frame frame in manifest.frames)
            RunFrame(frame, prefilter, blurH, blurV, upsample, paramBuffer, levels);

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

    static void RunFrame(Frame frame, ComputeShader prefilter, ComputeShader blurH,
                         ComputeShader blurV, ComputeShader upsample, ComputeBuffer paramBuffer,
                         int levels)
    {
        var input = new Texture2D(frame.width, frame.height, TextureFormat.RGBAFloat,
                                  mipChain: false, linear: true);
        input.SetPixelData(ReadFloats(Path.Combine(ParityDir, frame.name + "_input.f32"),
                                      frame.width * frame.height * 4), 0);
        input.Apply(updateMipmaps: false);

        Vector2Int[] sizes = MipSizes(frame.width, frame.height);
        if (sizes.Length - 1 != levels)
            throw new InvalidDataException($"{frame.name}: {sizes.Length - 1} upsample levels, weights have {levels}");

        int n = sizes.Length - 1;
        var mips = new RenderTexture[n + 1];
        var ups = new RenderTexture[n];        // ups[n] is mips[n]
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
                upsample.SetInt("_ParamOffset", k * ParamCount);
                Dispatch(upsample, mips[k], ups[k]);
            }

            for (int i = 0; i <= n; ++i)
                WriteFloats(mips[i], Path.Combine(ParityDir, $"{frame.name}_unity_mip{i}.f32"));
            for (int k = 0; k < n; ++k)
                WriteFloats(ups[k], Path.Combine(ParityDir, $"{frame.name}_unity_up{k}.f32"));
        }
        finally
        {
            foreach (RenderTexture rt in mips) if (rt != null) rt.Release();
            foreach (RenderTexture rt in ups) if (rt != null) rt.Release();
            UnityEngine.Object.DestroyImmediate(input);
        }
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

    static float[] LoadWeights()
    {
        byte[] bytes = AssetDatabase.LoadAssetAtPath<TextAsset>(WeightsPath).bytes;
        if (bytes.Length == 0 || bytes.Length % (ParamCount * sizeof(float)) != 0)
            throw new InvalidDataException($"{WeightsPath}: {bytes.Length} bytes is not a whole number of {ParamCount}-float levels");
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
