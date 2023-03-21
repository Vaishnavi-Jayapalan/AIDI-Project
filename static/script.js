const form = document.getElementById('upload-form');
const input = document.getElementById('file-input');
const button = document.getElementById('submit-button');
const resultContainer = document.getElementById('result-container');
const result = document.getElementById('result');
const image = document.getElementById('image');

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const formData = new FormData();
  formData.append('file', input.files[0]);

  button.disabled = true;
  resultContainer.classList.add('hidden');
  result.textContent = '';
  image.src = '';

  fetch('/', {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then((data) => {
      result.textContent = `Predicted
