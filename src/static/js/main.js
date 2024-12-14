document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const preview = document.getElementById('preview');
    const results = document.getElementById('results');
    const resultsBody = document.getElementById('resultsBody');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');

    // Preview image when selected
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                imagePreview.classList.remove('d-none');
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const file = imageInput.files[0];
        if (!file) {
            showError('الرجاء اختيار صورة');
            return;
        }

        // Show loading
        loading.classList.remove('d-none');
        results.classList.add('d-none');
        error.classList.add('d-none');

        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Send request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('حدث خطأ أثناء تحليل الصورة');
            }

            const data = await response.json();

            // Display results
            resultsBody.innerHTML = '';
            data.results.forEach(result => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${result.letter}</td>
                    <td>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${result.confidence * 100}%"
                                 aria-valuenow="${result.confidence * 100}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                                ${(result.confidence * 100).toFixed(2)}%
                            </div>
                        </div>
                    </td>
                `;
                resultsBody.appendChild(row);
            });

            results.classList.remove('d-none');
        } catch (err) {
            showError(err.message);
        } finally {
            loading.classList.add('d-none');
        }
    });

    function showError(message) {
        error.textContent = message;
        error.classList.remove('d-none');
    }
});
