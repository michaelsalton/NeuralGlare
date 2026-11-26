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
        public string rayStorage;                                   // "fp16": streak targets (rayA / rayB) RGBA16F; missing: fp32
                                                                    // (efficiency step 1a, decision log 2026-11-24; neuralglare.storage)

        public RenderTextureFormat RayFormat =>
            rayStorage == "fp16" ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGBFloat;
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
            if (!string.IsNullOrEmpty(layout.rayStorage) && layout.rayStorage != "fp16" && layout.rayStorage != "fp32")
                throw new InvalidDataException($"{source}: rayStorage \"{layout.rayStorage}\", expected fp16 or fp32");
            // glare.slang PASSES / PROFILE_COUNT: 3 passes of 4 taps reach 0..63 texels. A 2a layout
            // (streakTaps 8: 2 passes, 32 logits) fails here; JsonUtility ignores the key itself.
            if (layout.spikeParamCount > 0 && (layout.passes != 3 || layout.profileCount != 24))
                throw new InvalidDataException($"{source}: {layout.passes} passes and {layout.profileCount} profile logits, expected 3 and 24 (glare.slang)");
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
        // Optional: the ray-stack path (efficiency step 1c); without them the per-ray passes run.
        public ComputeShader streakToStack, spikeCompositeStack;
        public bool HasRayStack => streakToStack != null && spikeCompositeStack != null;
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
                            RayScaleId = Shader.PropertyToID("_RayScale"),
                            AccumId = Shader.PropertyToID("_Accum"),
                            SpikeEvenId = Shader.PropertyToID("_SpikeEven"),
                            SpikeOddId = Shader.PropertyToID("_SpikeOdd"),
                            RayStackId = Shader.PropertyToID("_RayStack"),
                            RayStackOutId = Shader.PropertyToID("_RayStackOut");

        readonly GlareShaders shaders;
        readonly GlareLayout layout;
        readonly ComputeBuffer parameters;
        Vector2Int size = new(-1, -1);
        RenderTexture[] mips = Array.Empty<RenderTexture>(), blurred = Array.Empty<RenderTexture>(),
                        ups = Array.Empty<RenderTexture>();
        RenderTexture full, rayA, rayB, spikes;
        RenderTexture rayStack;                                          // RayStack only
        readonly RenderTexture[][] families = new RenderTexture[2][];    // [family][ping-pong copy]

        // rayStack: use the ray-stack path where it applies (RayStack says whether it does).
        public GlareChain(GlareShaders shaders, GlareLayout layout, float[] weights, bool rayStack = true)
        {
            if (!shaders.Complete)
                throw new ArgumentException("GlareShaders: a compute shader is missing");
            if (weights.Length != layout.FloatCount)
                throw new ArgumentException($"{weights.Length} weights, the layout says {layout.FloatCount}");
            this.shaders = shaders;
            this.layout = layout;
            // Efficiency step 1c (decision log 2026-11-24): every ray's last streak pass into a slice
            // of one texture array, and one composite pass does the cross filter, brightness and
            // family sums (glare.slang spike_composite_stack). Needs brightness, and the rays at the
            // output's resolution (the composite reads the stack at its own texels).
            RayStack = rayStack && shaders.HasRayStack && layout.brightParamCount > 0
                       && layout.spikeSource == "input";
            parameters = new ComputeBuffer(weights.Length, sizeof(float));
            parameters.SetData(weights);
        }

        // Pre-exposure k of the rays (glare.slang streak; decision log 2026-11-24): the light is
        // scaled by k before the first streak pass stores it and by 1/k when a ray joins its family's
        // float32 sum, so fp16 streak targets hold a real sun. The output doesn't depend on k (exactly,
        // for a power of two). URP has no exposure to take it from, so it is a fixed power of two by
        // default; a game would set it per frame from its exposure. train_points.py RAY_PRE_EXPOSURE.
        public const float DefaultPreExposure = 1f / 16f;
        public float PreExposure { get; set; } = DefaultPreExposure;

        public bool RayStack { get; }                           // the ray-stack path (step 1c) runs
        public bool HasFullLevel { get; private set; }
        public bool HasSpikes => layout.spikeParamCount > 0;
        public RenderTexture[] Mips => mips;
        public RenderTexture[] Ups => ups;
        public RenderTexture Full => full;
        public RenderTexture FamilyEven { get; private set; }   // valid after Record; null with RayStack
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
                // Streak outputs in the layout's format (train_points.py stored() rounds the same);
                // family sums stay float32: with brightness they hold e^z times each ray before the
                // composite normalises, up to ~9x the source, past fp16's 65504
                rayA = Target(raySize, "rayA", layout.RayFormat);
                rayB = Target(raySize, "rayB", layout.RayFormat);
                if (RayStack)       // the last streak passes, one slice per ray; no family sums stored
                    rayStack = Target(raySize, "rayStack", layout.RayFormat, layout.rays);
                else
                {
                    families[0] = new[] { Target(raySize, "even0"), Target(raySize, "even1") };
                    families[1] = new[] { Target(raySize, "odd0"), Target(raySize, "odd1") };
                }
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
            {
                RenderTargetIdentifier raySource = layout.spikeSource == "input" ? source : mips[0];
                if (RayStack)
                    RecordRayStack(cmd, raySource);
                else
                    RecordRays(cmd, raySource);
            }

            using (new ProfilingScope(cmd, Composite))
            {
                if (RayStack)
                {
                    ComputeShader stacked = shaders.spikeCompositeStack;
                    cmd.SetComputeBufferParam(stacked, 0, ParamsId, parameters);
                    cmd.SetComputeTextureParam(stacked, 0, RayStackId, rayStack);
                    cmd.SetComputeIntParam(stacked, ParamOffsetId, layout.levels * layout.paramCount);
                    cmd.SetComputeFloatParam(stacked, RayScaleId, RayScale(PreExposure, false, true));
                    Dispatch(cmd, stacked, full, spikes);
                    return;
                }
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
                    cmd.SetComputeFloatParam(shader, RayScaleId,
                                             RayScale(PreExposure, stage == 0, accumulate));
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
                    cmd.SetComputeFloatParam(crossPass, RayScaleId, RayScale(PreExposure, false, true));
                    cmd.SetComputeTextureParam(crossPass, 0, AccumId, families[f][current[f]]);
                    RenderTexture next = families[f][1 - current[f]];
                    Dispatch(cmd, crossPass, last, next);
                    current[f] ^= 1;
                }
            }
            FamilyEven = families[0][current[0]];
            FamilyOdd = families[1][current[1]];
        }

        // The ray-stack path (step 1c): each ray's streak passes, the last one into slice `ray` of
        // rayStack. The cross filter, brightness and family sums happen in the composite.
        void RecordRayStack(CommandBuffer cmd, RenderTargetIdentifier source)
        {
            int profile = layout.levels * layout.paramCount + layout.spikeParamCount;
            ComputeShader streak = shaders.streak, toStack = shaders.streakToStack;
            foreach (ComputeShader shader in new[] { streak, toStack })
            {
                cmd.SetComputeBufferParam(shader, 0, ParamsId, parameters);
                cmd.SetComputeIntParam(shader, ParamOffsetId, profile);
            }
            cmd.SetComputeTextureParam(toStack, 0, RayStackOutId, rayStack);

            for (int ray = 0; ray < layout.rays; ++ray)
            {
                RenderTargetIdentifier s = source;
                RenderTexture last = null;
                for (int stage = 0; stage < layout.passes; ++stage)
                {
                    bool final = stage == layout.passes - 1;
                    ComputeShader shader = final ? toStack : streak;
                    cmd.SetComputeIntParam(shader, RayId, ray);
                    cmd.SetComputeIntParam(shader, StageId, stage);
                    cmd.SetComputeFloatParam(shader, RayScaleId, RayScale(PreExposure, stage == 0, false));
                    if (final)
                    {
                        // _Target is unused by StreakToStack; the group count comes from the stack
                        cmd.SetComputeTextureParam(shader, 0, SourceId, s);
                        cmd.DispatchCompute(shader, 0, (rayStack.width + 7) / 8, (rayStack.height + 7) / 8, 1);
                        break;
                    }
                    RenderTexture next = last == rayA ? rayB : rayA;
                    Dispatch(cmd, shader, s, next);
                    s = next;
                    last = next;
                }
            }
            FamilyEven = FamilyOdd = null;
        }

        // A ray pass's scale (glare.slang streak): k if it reads the source, 1/k if it writes the
        // family sum, both if it does both. As train_points.py ray_scale().
        static float RayScale(float k, bool readsSource, bool writesFamily) =>
            (readsSource ? k : 1f) * (writesFamily ? 1f / k : 1f);

        // Target size comes from _Target itself (GetDimensions in the kernels), 8x8 thread groups.
        static void Dispatch(CommandBuffer cmd, ComputeShader shader, RenderTargetIdentifier source,
                             RenderTexture target)
        {
            cmd.SetComputeTextureParam(shader, 0, SourceId, source);
            cmd.SetComputeTextureParam(shader, 0, TargetId, target);
            cmd.DispatchCompute(shader, 0, (target.width + 7) / 8, (target.height + 7) / 8, 1);
        }

        // slices > 0: a texture array of that many slices (the ray stack).
        static RenderTexture Target(Vector2Int size, string name,
                                    RenderTextureFormat format = RenderTextureFormat.ARGBFloat,
                                    int slices = 0)
        {
            if (!SystemInfo.SupportsRandomWriteOnRenderTextureFormat(format))
                throw new NotSupportedException($"{name}: {format} can't be a UAV on this GPU");
            var rt = new RenderTexture(size.x, size.y, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "NeuralGlare." + name,
                enableRandomWrite = true,
                useMipMap = false,
                filterMode = FilterMode.Point,
                hideFlags = HideFlags.HideAndDontSave,
            };
            if (slices > 0)
            {
                rt.dimension = TextureDimension.Tex2DArray;
                rt.volumeDepth = slices;
            }
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
            Free(full); Free(rayA); Free(rayB); Free(spikes); Free(rayStack);
            rayStack = null;
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
