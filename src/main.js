const { app, BrowserWindow } = require('electron');
if (require('electron-squirrel-startup')) app.quit();

const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

const { downloadPythonExecutable, createVirtualEnvironment } = require('./scripts/pythonSetup.js');

const URL = "http://127.0.0.1:35565";

let mainWindow;
let pythonBackend;

async function createWindow() {
  mainWindow = new BrowserWindow({
    show: false,
    backgroundColor: "#0f0f11",
    icon: process.platform === 'win32' ? path.join(__dirname, 'assets', 'logo.ico') : path.join(__dirname, 'assets', 'logo.png'),
    width: 1280,
    height: 720,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  
  await mainWindow.loadFile(path.join(__dirname, 'index.html'));
  mainWindow.removeMenu();
  mainWindow.show();
  global.mainWindow = mainWindow;
  
  try {
    const pythonRoot = app.isPackaged ? process.resourcesPath : path.resolve();
    const pythonBinDir = await downloadPythonExecutable(pythonRoot);
    const venvPath = await createVirtualEnvironment(pythonRoot, pythonBinDir);
    const pythonExecutable = path.join(venvPath, process.platform === 'win32' ? 'Scripts\\python.exe' : 'bin/python');
    const pythonScript = path.join(pythonRoot, 'backend', 'main.py');
    
    pythonBackend = spawn(pythonExecutable, [pythonScript]);
    pythonBackend.stdout.on('data', (data) => {
      console.log(`${data}`);
    });
    pythonBackend.stderr.on('data', (data) => {
      console.log(`${data}`);
    });
  } catch (error) {
    console.error('Failed to set up Python:', error);
    app.quit();
    return;
  }
  
  function waitForServer(url, maxTries = 20) {
    return new Promise((resolve, reject) => {
      function checkConnection(attempts) {
        if (attempts <= 0) {
          return reject(new Error('Server never responded'));
        }
  
        http.get(url, (res) => {
          resolve(true);
        }).on('error', () => {
          console.log('Server not responding, retrying...');
          setTimeout(() => checkConnection(attempts - 1), 750);
        });
      }
      
      checkConnection(maxTries);
    });
  }
  
  waitForServer(URL)
    .then(() => mainWindow.loadURL(URL))
    .catch((err) => {
      console.error(err);
      app.quit();
    });
}

const isFirstInstance = app.requestSingleInstanceLock();

if (isFirstInstance) {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  })
  
  app.on('ready', () => {
    createWindow();
  })
} else {
  app.quit()
}

app.on('activate', function () {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (pythonBackend) {
      pythonBackend.kill();
    }
    app.quit();
  }
});
