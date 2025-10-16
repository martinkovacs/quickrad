const { FusesPlugin } = require('@electron-forge/plugin-fuses');
const { FuseV1Options, FuseVersion } = require('@electron/fuses');
const path = require('path');
const fs = require('fs');

module.exports = {
  packagerConfig: {
    asar: true,
    ignore: ["/backend$"],
    icon: "./assets/logo",
  },
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {
        name: 'QuickRad',
        setupExe: 'quickrad.exe',
        setupIcon: './assets/logo.ico',
      },
      platforms: ["win32"],
    },
    {
      name: '@electron-forge/maker-zip',
      config: {
        options: {
          icon: './assets/logo.png',
        }
      },
      platforms: ["linux"]
    }
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {},
    },
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ],
  hooks: {
    packageAfterCopy: (forgeConfig, buildPath, electronPlatformName, arch) => {
      const sourcePath = path.join(path.resolve(), 'backend');
      const destPath = path.join(buildPath, '..', 'backend');
      
      fs.cpSync(sourcePath, destPath, {
        force: true,
        recursive: true,
        filter: (src) => {
          const relativePath = path.relative(sourcePath, src);
          
          if (relativePath.endsWith("python") || relativePath.endsWith("venv") || relativePath.includes("viewer-iframe-")) return false;

          return true;
        }
      });
    }
  }
};
