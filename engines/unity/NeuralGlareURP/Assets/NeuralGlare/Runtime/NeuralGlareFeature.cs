// URP renderer feature: runs GlareChain on the camera colour before post-processing (Phase 3).
// Add it to the URP Renderer asset and assign the 10 compute shaders, glare_params.bytes and
// glare_params.json (written by unity_parity.py export). The camera must render HDR (URP asset).
// -ngWeights <file.bytes> -ngLayout <file.json> on the command line replace the assigned weights,
// so one build can time any model that runs on the current kernels.
using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace NeuralGlare
{
    public sealed class NeuralGlareFeature : ScriptableRendererFeature
    {
        [SerializeField] GlareShaders shaders = new();
        [SerializeField] TextAsset weights;     // Data/glare_params.bytes
        [SerializeField] TextAsset layout;      // Data/glare_params.json
        [SerializeField] RenderPassEvent passEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        GlareChain chain;
        GlarePass pass;

        // The weights actually loaded (GlareTimer records it): the assets assigned here, or the files
        // named by -ngWeights / -ngLayout (both or neither).
        public static string WeightsSource { get; private set; }

        public override void Create()
        {
            chain?.Dispose();
            chain = null;
            string weightsPath = GlareTimer.Arg("-ngWeights"), layoutPath = GlareTimer.Arg("-ngLayout");
            if ((weightsPath == null) != (layoutPath == null))
                throw new ArgumentException("NeuralGlareFeature: give -ngWeights and -ngLayout together");
            bool fromFiles = weightsPath != null;
            if (!shaders.Complete || (!fromFiles && (weights == null || layout == null)))
            {
                Debug.LogWarning("NeuralGlareFeature: assign the compute shaders, weights and layout");
                return;
            }
            string layoutName = fromFiles ? layoutPath : layout.name;
            string weightsName = fromFiles ? weightsPath : weights.name;
            GlareLayout l = GlareLayout.Parse(fromFiles ? File.ReadAllText(layoutPath) : layout.text, layoutName);
            float[] w = l.ReadWeights(fromFiles ? File.ReadAllBytes(weightsPath) : weights.bytes, weightsName);
            chain = new GlareChain(shaders, l, w);
            WeightsSource = weightsName;
            Debug.Log($"NeuralGlareFeature: weights {weightsName}");
            pass = new GlarePass(chain) { renderPassEvent = passEvent };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (chain != null && renderingData.cameraData.cameraType == CameraType.Game)
                renderer.EnqueuePass(pass);
        }

        protected override void Dispose(bool disposing)
        {
            chain?.Dispose();
            chain = null;
        }
    }

    sealed class GlarePass : ScriptableRenderPass
    {
        class PassData
        {
            public GlareChain chain;
            public TextureHandle color;
        }

        readonly GlareChain chain;

        public GlarePass(GlareChain chain)
        {
            this.chain = chain;
            requiresIntermediateTexture = true;     // read the camera colour, never the back buffer
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resources = frameData.Get<UniversalResourceData>();
            if (resources.isActiveTargetBackBuffer)
                return;
            TextureHandle color = resources.activeColorTexture;
            TextureDesc desc = renderGraph.GetTextureDesc(color);
            chain.Resize(desc.width, desc.height);

            // An unsafe pass: GlareChain records its dispatches into one native CommandBuffer on its
            // own persistent targets, so RenderGraph only needs to see the camera colour.
            using IUnsafeRenderGraphBuilder builder =
                renderGraph.AddUnsafePass<PassData>("NeuralGlare", out PassData data);
            data.chain = chain;
            data.color = color;
            builder.UseTexture(color, AccessFlags.ReadWrite);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext ctx) =>
            {
                CommandBuffer cmd = CommandBufferHelpers.GetNativeCommandBuffer(ctx.cmd);
                d.chain.Record(cmd, d.color);
                cmd.Blit(d.chain.Output, d.color);   // back into the camera colour, outside the timed scopes
            });
        }
    }
}
