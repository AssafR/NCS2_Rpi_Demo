function setDevice(device) {
  fetch('/device?name=' + device)
    .then(response => response.text())
    .then(text => console.log(text));
}
