// The glare model's dispatch sequence (slang/engine_glare.slang kernels), recorded into a
// CommandBuffer. One copy, used by the URP renderer feature (NeuralGlareFeature) and by the parity
// harness (Editor/ParityHarness.cs), so the code that is timed is the code parity checks.
// Order as train_points.py glare(): downsample chain, upsample levels, full-resolution level, then
// the spike term (each ray's streak passes, summed per family, composite).
// Targets are allocated once per resolution (Resize), never per frame.
using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace NeuralGlare
{
    // As unity_parity.py UNITY_LAYOUT (glare_params.json); JsonUtility fills the fields by name.
    [Serializable]
    public class GlareLayout
    {
        public int inputs, hidden, outputs, paramCount, levels;
        public int mips;                                            // the pyramid's mip count
        public int spikeParamCount, profileCount, rays, passes;     // 0 without the spike term
        public string spikeSource;                                  // "mip0" or "input"
        public int crossParamCount, crossTaps;                      // 0 without the cross filter
        public int brightParamCount;                                // 0 without per-ray brightness

        public int SpikeFloats => spikeParamCount > 0 ? spikeParamCount + profileCount : 0;
        public int FloatCount => levels * paramCount + SpikeFloats + crossParamCount + brightParamCount;

        public static GlareLayout Parse(string json, string source)
        {
            var layout = JsonUtility.FromJson<GlareLayout>(json);
            if (layout.paramCount <= 0 || layout.levels <= 0)
                throw new InvalidDataException($"{source}: paramCount {layout.paramCount}, levels {layout.levels}");
            if (layout.spikeParamCount > 0 && (layout.profileCount <= 0 || layout.rays <= 0 || layout.passes <= 0))
                throw new InvalidDataException($"{source}: spike section without profileCount / rays / passes");
            if (layout.spikeParamCount > 0 && layout.spikeSource != "mip0" && layout.spikeSource != "input")
                throw new InvalidDataException($"{source}: spikeSource \"{layout.spikeSource}\", expected mip0 or input");
            if (layout.mips <= 0)   // JsonUtility leaves a missing field 0: an old layout fails here, not as a 6-mip run
                throw new InvalidDataException($"{source}: mips {layout.mips} (0 mips isn't supported here; an old layout? rerun unity_parity.py export)");
            if (layout.crossParamCount > 0 && layout.spikeParamCount <= 0)
                throw new InvalidDataException($"{source}: cross filter without the spike term");
            if (layout.brightParamCount > 0 && layout.crossParamCount <= 0)
                throw new InvalidDataException($"{source}: per-ray brightness without the cross filter");
            return layout;
        }

        public float[] ReadWeights(byte[] bytes, string source)
        {
            if (bytes.Length != FloatCount * sizeof(float))
                throw new InvalidDataException($"{source}: {bytes.Length} bytes, the layout says {levels} levels of {paramCount} floats + {SpikeFloats} spike + {crossParamCount} cross + {brightParamCount} brightness floats");
            var weights = new float[FloatCount];
            Buffer.BlockCopy(bytes, 0, weights, 0, bytes.Length);
            return weights;
        }
    }

    // One kernel per .compute (Shaders/Glare*.compute), so every kernel index is 0.
    [Serializable]
    public class GlareShaders
    {
        public ComputeShader prefilter, blurH, blurV, upsample, streak, streakAccumulate,
                             rayCrossAccumulate, rayBrightAccumulate, spikeComposite,
                             spikeCompositeBright;

        public bool Complete => prefilter != null && blurH != null && blurV != null && upsample != null
                                && streak != null && streakAccumulate != null && rayCrossAccumulate != null
                                && rayBrightAccumulate != null && spikeComposite != null
                                && spikeCompositeBright != null;
    }

    public sealed class GlareChain : IDisposable
    {
        // GPU timing scopes (GlareTimer reads them). Nested: Total holds the other four.
        public static readonly ProfilingSampler Total = new("NeuralGlare");
        public static readonly ProfilingSampler Pyramid = new("NeuralGlare.Pyramid");
        public static readonly ProfilingSampler Levels = new("NeuralGlare.Levels");
        public static readonly ProfilingSampler Rays = new("NeuralGlare.Rays");
        public static readonly ProfilingSampler Composite = new("NeuralGlare.Composite");
        public static Vector2Int LastSize { get; private set; }    // the resolution last recorded

        static readonly int SourceId = Shader.PropertyToID("_Source"),
                            TargetId = Shader.PropertyToID("_Target"),
                            SourceLowId = Shader.PropertyToID("_SourceLow"),
                            ParamsId = Shader.PropertyToID("_GlareParams"),
                            ParamOffsetId = Shader.PropertyToID("_ParamOffset"),
                            RayId = Shader.PropertyToID("_Ray"),
                            StageId = Shader.PropertyToID("_Stage"),
                            AccumId = Shader.PropertyToID("_Accum"),
                            SpikeEvenId = Shader.PropertyToID("_SpikeEven"),
                            SpikeOddId = Shader.PropertyToID("_SpikeOdd");

        readonly GlareShaders shaders;
        readonly GlareLayout layout;
        readonly ComputeBuffer parameters;
        Vector2Int size = new(-1, -1);
        RenderTexture[] mips = Array.Empty<RenderTexture>(), blurred = Array.Empty<RenderTexture>(),
                        ups = Array.Empty<RenderTexture>();
        RenderTexture full, rayA, rayB, spikes;
        readonly RenderTexture[][] families = new RenderTexture[2][];    // [family][ping-pong copy]

        public GlareChain(GlareShaders shaders, GlareLayout layout, float[] weights)
        {
            if (!shaders.Complete)
                throw new ArgumentException("GlareShaders: a compute shader is missing");
            if (weights.Length != layout.FloatCount)
                throw new ArgumentException($"{weights.Length} weights, the layout says {layout.FloatCount}");
            this.shaders = shaders;
            this.layout = layout;
            parameters = new ComputeBuffer(weights.Length, sizeof(float));
            parameters.SetData(weights);
        }

        public bool HasFullLevel { get; private set; }
        public bool HasSpikes => layout.spikeParamCount > 0;
        public RenderTexture[] Mips => mips;
        public RenderTexture[] Ups => ups;
        public RenderTexture Full => full;
        public RenderTexture FamilyEven { get; private set; }   // valid after Record
        public RenderTexture FamilyOdd { get; private set; }
        public RenderTexture Output => HasSpikes ? spikes : HasFullLevel ? full : ups[0];

        public void Resize(int width, int height)
        {
            if (size.x == width && size.y == height)
                return;
            ReleaseTargets();
            Vector2Int[] sizes = MipSizes(width, height, layout.mips);
            int n = sizes.Length - 1;
            // Phase 1 weights: one network per pyramid level. Phase 2: one more, the full-resolution
            // level whose "mip" is the input frame (train_points.py glare()).
            HasFullLevel = layout.levels == n + 1;
            if (layout.levels != n && !HasFullLevel)
                throw new InvalidOperationException($"{width}x{height}: {n} upsample levels, weights have {layout.levels}");
            if (HasSpikes && !HasFullLevel)
                throw new InvalidOperationException("spike weights without the full-resolution level");
            size = new Vector2Int(width, height);

            mips = new RenderTexture[n + 1];
            blurred = new RenderTexture[n + 1];                  // blurred[0] unused
            for (int i = 0; i <= n; ++i)
            {
                mips[i] = Target(sizes[i], $"mip{i}");
                if (i > 0)
                    blurred[i] = Target(sizes[i], $"blurH{i}");
            }
            ups = new RenderTexture[n];
            for (int k = 0; k < n; ++k)
                ups[k] = Target(sizes[k], $"up{k}");
            if (HasFullLevel)
                full = Target(size, "full");
            if (HasSpikes)
            {
                Vector2Int raySize = layout.spikeSource == "input" ? size : sizes[0];
                rayA = Target(raySize, "rayA");
                rayB = Target(raySize, "rayB");
                families[0] = new[] { Target(raySize, "even0"), Target(raySize, "even1") };
                families[1] = new[] { Target(raySize, "odd0"), Target(raySize, "odd1") };
                spikes = Target(size, "spikes");
            }
        }

        // source: the HDR input frame, at the size given to Resize.
        public void Record(CommandBuffer cmd, RenderTargetIdentifier source)
        {
            if (size.x < 0)
                throw new InvalidOperationException("GlareChain: Resize before Record");
            LastSize = size;
            int n = mips.Length - 1;
            using var total = new ProfilingScope(cmd, Total);

            // Downsample: mip 0 = prefilter, mip i = BlurV(BlurH(mip i-1)) at half the size
            using (new ProfilingScope(cmd, Pyramid))
            {
                Dispatch(cmd, shaders.prefilter, source, mips[0]);
                for (int i = 1; i <= n; ++i)
                {
                    Dispatch(cmd, shaders.blurH, mips[i - 1], blurred[i]);
                    Dispatch(cmd, shaders.blurV, blurred[i], mips[i]);
                }
            }

            // Upsample: level k reads mip k and level k + 1's output; then the full-resolution level
            // reads the input frame and up[0] (mip 0 itself with 1 mip)
            using (new ProfilingScope(cmd, Levels))
            {
                ComputeShader up = shaders.upsample;
                cmd.SetComputeBufferParam(up, 0, ParamsId, parameters);
                for (int k = n - 1; k >= 0; --k)
                {
                    cmd.SetComputeTextureParam(up, 0, SourceLowId, k + 1 < n ? ups[k + 1] : mips[n]);
                    cmd.SetComputeIntParam(up, ParamOffsetId, k * layout.paramCount);
                    Dispatch(cmd, up, mips[k], ups[k]);
                }
                if (HasFullLevel)
                {
                    cmd.SetComputeTextureParam(up, 0, SourceLowId, n > 0 ? ups[0] : mips[0]);
                    cmd.SetComputeIntParam(up, ParamOffsetId, n * layout.paramCount);
                    Dispatch(cmd, up, source, full);
                }
            }

            if (!HasSpikes)
                return;
            using (new ProfilingScope(cmd, Rays))
                RecordRays(cmd, layout.spikeSource == "input" ? source : mips[0]);

            using (new ProfilingScope(cmd, Composite))
            {
                ComputeShader composite = layout.brightParamCount > 0 ? shaders.spikeCompositeBright
                                                                      : shaders.spikeComposite;
                cmd.SetComputeBufferParam(composite, 0, ParamsId, parameters);
                cmd.SetComputeTextureParam(composite, 0, SpikeEvenId, FamilyEven);
                cmd.SetComputeTextureParam(composite, 0, SpikeOddId, FamilyOdd);
                cmd.SetComputeIntParam(composite, ParamOffsetId, layout.levels * layout.paramCount);
                Dispatch(cmd, composite, full, spikes);
            }
        }

        // Each ray: its streak passes from source, then added to its family's sum (ray k is in family
        // k % 2), as train_points.py spikes(). With the cross filter (decision log 2026-11-19) every
        // streak pass is plain and RayCrossAccumulate (RayBrightAccumulate with per-ray brightness)
        // does the adding. A dispatch can't read _Accum and write _Target on one texture, so each
        // family's sum ping-pongs between two copies.
        void RecordRays(CommandBuffer cmd, RenderTargetIdentifier source)
        {
            int profile = layout.levels * layout.paramCount + layout.spikeParamCount;
            bool cross = layout.crossParamCount > 0;
            ComputeShader crossPass = layout.brightParamCount > 0 ? shaders.rayBrightAccumulate
                                                                  : shaders.rayCrossAccumulate;
            var current = new int[2];                   // which copy holds each family's sum
            foreach (RenderTexture rt in new[] { families[0][0], families[1][0] })
            {
                cmd.SetRenderTarget(rt);
                cmd.ClearRenderTarget(false, true, Color.clear);
            }
            foreach (ComputeShader shader in new[] { shaders.streak, shaders.streakAccumulate })
            {
                cmd.SetComputeBufferParam(shader, 0, ParamsId, parameters);
                cmd.SetComputeIntParam(shader, ParamOffsetId, profile);
            }
            if (cross)
            {
                cmd.SetComputeBufferParam(crossPass, 0, ParamsId, parameters);
                cmd.SetComputeIntParam(crossPass, ParamOffsetId, profile + layout.profileCount);
            }

            for (int ray = 0; ray < layout.rays; ++ray)
            {
                int f = ray % 2;
                RenderTargetIdentifier s = source;
                RenderTexture last = null;
                for (int stage = 0; stage < layout.passes; ++stage)
                {
                    bool accumulate = !cross && stage == layout.passes - 1;
                    ComputeShader shader = accumulate ? shaders.streakAccumulate : shaders.streak;
                    cmd.SetComputeIntParam(shader, RayId, ray);
                    cmd.SetComputeIntParam(shader, StageId, stage);
                    RenderTexture next;
                    if (accumulate)
                    {
                        cmd.SetComputeTextureParam(shader, 0, AccumId, families[f][current[f]]);
                        next = families[f][1 - current[f]];
                        current[f] ^= 1;
                    }
                    else
                        next = last == rayA ? rayB : rayA;
                    Dispatch(cmd, shader, s, next);
                    s = next;
                    last = next;
                }
                if (cross)
                {
                    cmd.SetComputeIntParam(crossPass, RayId, ray);
                    cmd.SetComputeTextureParam(crossPass, 0, AccumId, families[f][current[f]]);
                    RenderTexture next = families[f][1 - current[f]];
                    Dispatch(cmd, crossPass, last, next);
                    current[f] ^= 1;
                }
            }
            FamilyEven = families[0][current[0]];
            FamilyOdd = families[1][current[1]];
        }

        // Target size comes from _Target itself (GetDimensions in the kernels), 8x8 thread groups.
        static void Dispatch(CommandBuffer cmd, ComputeShader shader, RenderTargetIdentifier source,
                             RenderTexture target)
        {
            cmd.SetComputeTextureParam(shader, 0, SourceId, source);
            cmd.SetComputeTextureParam(shader, 0, TargetId, target);
            cmd.DispatchCompute(shader, 0, (target.width + 7) / 8, (target.height + 7) / 8, 1);
        }

        static RenderTexture Target(Vector2Int size, string name)
        {
            var rt = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.ARGBFloat,
                                       RenderTextureReadWrite.Linear)
            {
                name = "NeuralGlare." + name,
                enableRandomWrite = true,
                useMipMap = false,
                filterMode = FilterMode.Point,
                hideFlags = HideFlags.HideAndDontSave,
            };
            rt.Create();
            return rt;
        }

        // As mip_sizes() in pyramid_bloom.py (URP Bloom, downscale "half"). Integer log2, so
        // powers of two can't round down the way Mathf.Log(x, 2) can.
        static Vector2Int[] MipSizes(int width, int height, int maxIterations)
        {
            int w = width >> 1, h = height >> 1;
            int count = Mathf.Clamp(FloorLog2(Mathf.Max(w, h)) - 1, 1, maxIterations);
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

        static void Free(RenderTexture rt)
        {
            if (rt == null)
                return;
            rt.Release();
            CoreUtils.Destroy(rt);
        }

        void ReleaseTargets()
        {
            foreach (RenderTexture rt in mips) Free(rt);
            foreach (RenderTexture rt in blurred) Free(rt);
            foreach (RenderTexture rt in ups) Free(rt);
            Free(full); Free(rayA); Free(rayB); Free(spikes);
            foreach (RenderTexture[] pair in families)
                if (pair != null)
                    foreach (RenderTexture rt in pair) Free(rt);
            mips = blurred = ups = Array.Empty<RenderTexture>();
            full = rayA = rayB = spikes = null;
            families[0] = families[1] = null;
            FamilyEven = FamilyOdd = null;
            size = new Vector2Int(-1, -1);
        }

        public void Dispose()
        {
            ReleaseTargets();
            parameters.Release();
        }
    }
}
