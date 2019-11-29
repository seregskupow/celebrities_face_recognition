const imageUpload = document.getElementById("imageUpload");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models")
]).then(start);

async function start() {
  const container = document.createElement("div");
  const res_cont = document.querySelector(".reco");
  container.style.position = "relative";
  document.querySelector(".result").append(container);
  // const labeledFaceDescriptors = await loadLabeledImages();
  // localStorage.setItem("data",JSON.stringify(labeledFaceDescriptors))
  let f = await JSON.parse(localStorage.getItem("data"));
  let a = [];
  await f.map((item, index) => {
    let arr = [];
    arr[0] = new Float32Array(Object.keys(item._descriptors[0]).length);
    arr[1] = new Float32Array(Object.keys(item._descriptors[1]).length);
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < Object.keys(item._descriptors[j]).length; i++) {
        arr[j][i] = item._descriptors[j][i];
      }
    }
    a[index] = new faceapi.LabeledFaceDescriptors(item._label, arr);
  });
  console.log(a);
  const faceMatcher = new faceapi.FaceMatcher(a, 0.6);
  let image;
  let canvas;
  document.querySelector(".loader").style.display = "none";
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map(d =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    res_cont.innerHTML = "";
    res_cont.style.display = "block";
    results.forEach((result, i) => {
      let a = result.toString().split(" ");
      let name = a.slice(0, 2).join(" ");
      if (!name.includes("unknown")) {
        res_cont.innerHTML += `<div><a href="https://en.wikipedia.org/wiki/${name}">${name}</a></div>`;
      }

      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString()
      });
      drawBox.draw(canvas);
    });
  });
}

function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Tony Stark"
  ];
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      console.log(descriptions);
      console.log(new faceapi.LabeledFaceDescriptors(label, descriptions));
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
