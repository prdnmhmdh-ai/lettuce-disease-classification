document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const captureButton = document.getElementById("capture");
  const cameraInput = document.getElementById("camera_image");
  const preview = document.getElementById("preview-camera");

  // Start camera
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
        video.play();
      });
  }

  // Capture photo and preview
  captureButton.addEventListener("click", () => {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
    cameraInput.value = dataUrl;

    // Show preview
    preview.innerHTML = '';
    const img = document.createElement("img");
    img.src = dataUrl;
    img.style.maxHeight = "300px";
    preview.appendChild(img);
  });
});
