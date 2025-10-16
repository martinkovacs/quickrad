const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onInstallProgress: (callback) => ipcRenderer.on('install-progress', (event, text) => callback(text))
});
