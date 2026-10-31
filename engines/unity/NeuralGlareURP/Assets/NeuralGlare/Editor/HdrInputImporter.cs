// HDR inputs must reach the shaders bit-exact: float32, uncompressed, linear, no mips.
// Settings live here rather than in .meta files so they survive a fresh copy of the data
// (Assets/Data/Inputs/ is gitignored and filled from data/inputs/).
using UnityEditor;
using UnityEngine;

class HdrInputImporter : AssetPostprocessor
{
    const string InputFolder = "Assets/Data/Inputs/";

    void OnPreprocessTexture()
    {
        if (!assetPath.StartsWith(InputFolder) || !assetPath.EndsWith(".exr"))
            return;

        var importer = (TextureImporter)assetImporter;
        importer.textureType = TextureImporterType.Default;
        importer.sRGBTexture = false;
        importer.mipmapEnabled = false;
        importer.npotScale = TextureImporterNPOTScale.None;
        importer.filterMode = FilterMode.Point;
        importer.wrapMode = TextureWrapMode.Clamp;
        importer.textureCompression = TextureImporterCompression.Uncompressed;
        importer.maxTextureSize = 8192;

        // Default platform: the default (Automatic) format would be BC6H on PC, which is lossy.
        var settings = importer.GetDefaultPlatformTextureSettings();
        settings.format = TextureImporterFormat.RGBAFloat;
        settings.textureCompression = TextureImporterCompression.Uncompressed;
        settings.maxTextureSize = 8192;
        importer.SetPlatformTextureSettings(settings);
    }
}
