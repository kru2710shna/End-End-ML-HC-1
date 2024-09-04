document.querySelector("form").addEventListener("submit", function(event) {
    const symptomsInput = document.querySelector("textarea[name='symptoms']");
    if (symptomsInput.value.trim() === "") {
        event.preventDefault();
        alert("Please enter at least one symptom.");
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const loadingScreen = document.querySelector('.loading-screen');
    const content = document.querySelector('.content');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        // Show loading screen and blur content
        loadingScreen.classList.add('active');
        content.classList.add('content-blur');

        // Fixed loading duration of 4 seconds
        setTimeout(() => {
            // Hide loading screen and remove blur effect
            loadingScreen.classList.remove('active');
            content.classList.remove('content-blur');

            // Submit the form
            form.submit();
        }, 3000); // 4000 milliseconds = 4 seconds
    });
});

function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const resultBox = document.querySelector('.diagnosis-results');
    if (resultBox) {
        const resultContent = resultBox.innerText;
        doc.text(resultContent, 10, 10);
        doc.save('prescription.pdf');3
    } else {
        alert('No diagnosis results to download.');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const downloadPdfButton = document.getElementById('download-pdf');
    if (downloadPdfButton) {
        downloadPdfButton.addEventListener('click', downloadPDF);
    }
});

function scrollToForm() {
    const formSection = document.getElementById('symptom-form-section');
    formSection.scrollIntoView({ behavior: 'smooth' });
}
