const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadForm = document.getElementById("upload-form");
const uploadBtn = document.getElementById("upload-btn");

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadForm.submit();
});

uploadBtn.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    uploadForm.submit();
  }
});
