// Request a device switch (CPU or MYRIAD). The Python server does the real work.
function setDevice(device) {
  fetch('/device?name=' + device)
    .then(response => response.text())
    .then(text => console.log(text));
}
