import { exec } from 'child_process';
import fs from 'fs';
import https from 'https';
import path from 'path';

import * as tar from 'tar';

const PYTHON_DOWNLOADS = {
  linux: 'https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.12.10+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz',
  win32: 'https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.12.10+20250409-x86_64-pc-windows-msvc-install_only_stripped.tar.gz',
  darwin: 'https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.12.10+20250409-aarch64-apple-darwin-install_only_stripped.tar.gz'
};

const PYTORCH_INDEX_URLS = {
  0: "https://download.pytorch.org/whl/cpu",
  118: "https://download.pytorch.org/whl/cu118",
  126: "https://download.pytorch.org/whl/cu126",
  128: "https://download.pytorch.org/whl/cu128",
}

function downloadFile(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        return downloadFile(response.headers.location).then(resolve).catch(reject);
      }

      if (response.statusCode !== 200) {
        return reject(new Error(`Failed to download: ${response.statusCode}`));
      }
      
      resolve(response);
    }).on('error', (err) => reject(err));
  });
}

export async function downloadPythonExecutable(pythonRoot) {
  const platform = process.platform
  const pythonUrl = PYTHON_DOWNLOADS[platform];
  
  if (!pythonUrl) {
    throw new Error(`Unsupported platform: ${platform}`);
  }

  const pythonExtractDir = path.join(pythonRoot, 'backend');
  const downloadPath = path.join(pythonExtractDir, 'download.tar.gz');
  const pythonBinDir = path.join(pythonExtractDir, 'python')

  if (fs.existsSync(pythonBinDir)) {
    console.log('Python binary directory already exists');
    return pythonBinDir;
  }
  
  fs.mkdirSync(pythonExtractDir, { recursive: true });
  
  global.mainWindow.webContents.send('install-progress', `Downloading python...`);
  const response = await downloadFile(pythonUrl, downloadPath);
  
  await new Promise((resolve, reject) => {
    const writeStream = fs.createWriteStream(downloadPath);
    response.pipe(writeStream);
    writeStream.on('finish', resolve);
    writeStream.on('error', reject);
  });

  await tar.x({
    file: downloadPath,
    cwd: pythonExtractDir
  })

  fs.unlinkSync(downloadPath);
  
  return pythonBinDir;
}

export async function createVirtualEnvironment(pythonRoot, pythonBinDir) {
  const platform = process.platform
  const venvPath = path.join(pythonRoot, 'backend', 'venv');
  const requirementsPath = path.join(pythonRoot, 'backend', 'requirements.txt');
  
  if (fs.existsSync(venvPath)) {
    console.log('Virtual environment already exists');
    return path.join(venvPath);
  }

  global.mainWindow.webContents.send('install-progress', `Creating virtual environment...`);

  const pythonExecutable = platform === 'win32' 
    ? path.join(pythonBinDir, 'python.exe')
    : path.join(pythonBinDir, 'bin/python3');
  await new Promise((resolve, reject) => {
    exec(`${pythonExecutable} -m venv ${venvPath}`, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
  
  const pipExecutable = platform === 'win32'
    ? path.join(venvPath, 'Scripts', 'pip.exe')
    : path.join(venvPath, 'bin', 'pip');
  const dependencies = fs.readFileSync(requirementsPath, 'utf-8')
    .split('\n')
    .filter((line) => line.trim() && !line.startsWith('#'));
    
  // TODO: Refactor: if there is no pytorch dependency run in one command, otherwise run the torch dependencies separately
  for (const dependency of dependencies) {
    let counter = `(${dependencies.indexOf(dependency) + 1}/${dependencies.length})`;
    if (/^(torch|torchvision|torchaudio)/.test(dependency)) {
      let cudaVersion = await getCudaVersion();
      await installWithPip(pipExecutable,`${dependency} --index-url ${PYTORCH_INDEX_URLS[getPytorchIndexUrlKey(cudaVersion)]}`, counter);
      // Commented out, because RTX 50 series minimum CUDA version is 12.8, and on linux the default version is 12.6
      // if ((platform === 'darwin') || (platform === 'linux' && cudaVersion >= 126) || (platform === 'win32' && cudaVersion === 0)) {
      //   // Default pip package
      //   await installWithPip(pipExecutable, dependency, counter)
      // } else {
      //   // Custom index url
      //   await installWithPip(pipExecutable,`${dependency} --index-url ${PYTORCH_INDEX_URLS[getPytorchIndexUrlKey(cudaVersion)]}`, counter);
      // }
    } else {
      await installWithPip(pipExecutable, dependency, counter);
    }
  }

  return venvPath;
}

function getCudaVersion() {
  return new Promise((resolve, reject) => {
    exec('nvidia-smi', (error, stdout, stderr) => {
      if (error) {
        resolve(0);
        return;
      }

      const match = stdout.match(/CUDA Version:\s*(\d+)\.(\d+)/);
      if (!match) {
        resolve(0);
        return;
      }

      resolve(Number(match[1] + match[2]));
    });
  });
};

function getPytorchIndexUrlKey(cuda_version) {
  return Math.max(...Object.keys(PYTORCH_INDEX_URLS)
      .filter(key => Number(key) <= cuda_version)
      .map(Number));
};

async function installWithPip(pipExecutable, packageString, counter) {
  return new Promise(async (resolve, reject) => {
    global.mainWindow.webContents.send('install-progress', `Installing ${packageString.split("==")[0].split(" ")[0]}... ${counter}`);
    exec(`${pipExecutable} install ${packageString}`, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
};
