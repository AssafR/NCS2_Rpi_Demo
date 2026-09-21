// Request a device switch (CPU or MYRIAD). The Python server does the real work.
function setDevice(device) {
  fetch('/device?name=' + device)
    .then(response => response.text())
    .then(text => console.log(text));
}

// The camera image is simple by default. This button asks Python to add or
// remove the heatmap grid from the streamed image.
let heatmapsVisible = false;

function toggleHeatmaps() {
  heatmapsVisible = !heatmapsVisible;
  const button = document.getElementById('heatmap-button');

  fetch('/heatmaps?show=' + (heatmapsVisible ? '1' : '0'))
    .then(response => response.text())
    .then(text => {
      console.log(text);
      button.textContent = heatmapsVisible ? 'Hide heatmaps' : 'Show heatmaps';
    });
}
