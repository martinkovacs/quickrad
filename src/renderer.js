window.electronAPI.onInstallProgress((text) => {
    document.getElementById('install-progress').innerText = text;
});
