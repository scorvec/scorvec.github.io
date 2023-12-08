const fs = require('fs');
const path = require('path');

const currentDirectory = process.cwd();
const imageExtensions = ['.png'];

function getImagesInDirectory(directory) {
  const subdirectories = fs.readdirSync(directory, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  const imagesBySubdirectory = {};

  subdirectories.forEach(subdirectory => {
    const subdirectoryPath = path.join(directory, subdirectory);
    const images = fs.readdirSync(subdirectoryPath)
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return imageExtensions.includes(ext);
      })
      .map(file => path.join(subdirectoryPath, file));

    imagesBySubdirectory[subdirectory] = images;
  });

  return imagesBySubdirectory;
}

function generateHTML(imagesBySubdirectory) {
  const styles = `
    <style>
      /* Your existing styles remain unchanged */

      #subdirectory-dropdown {
        margin-bottom: 10px;
      }
    </style>
  `;

  const subdirectories = Object.keys(imagesBySubdirectory);

  const htmlContent = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Image Animation</title>
      ${styles}
    </head>
    <body>

    <!-- Add the subdirectory dropdown -->
    <label for="subdirectory-dropdown">Select Subdirectory:</label>
    <select id="subdirectory-dropdown">
      ${subdirectories.map(subdirectory => `<option value="${subdirectory}">${subdirectory}</option>`).join('')}
    </select>

    <div id="image-container">
      <!-- Images will be dynamically loaded here -->
    </div>

    <div id="control-container">
      <button onclick="changeSpeed(1)">Increase Speed</button>
      <button onclick="changeSpeed(-1)">Decrease Speed</button>
      <button onclick="prevImage()">Previous</button>
      <button onclick="togglePlayPause()">Play/Pause</button>
      <button onclick="nextImage()">Next</button>
    </div>

    <script>
      const imagesBySubdirectory = ${JSON.stringify(imagesBySubdirectory)};
      let currentIndex = 0;
      let intervalId;
      let speed = 1000; // Default speed in milliseconds

      // Populate the subdirectory dropdown dynamically
      const subdirectoryDropdown = document.getElementById('subdirectory-dropdown');

      // Event listener for the subdirectory dropdown
      subdirectoryDropdown.addEventListener('change', function () {
        const selectedSubdirectory = subdirectoryDropdown.value;
        currentIndex = 0;
        loadImages(imagesBySubdirectory[selectedSubdirectory]);
      });

      function loadImages(images) {
        const imageContainer = document.getElementById('image-container');
        imageContainer.innerHTML = ""; // Clear previous images
        const img = document.createElement('img');
        img.src = images[currentIndex];
        imageContainer.appendChild(img);
      }

      function nextImage() {
        const selectedSubdirectory = subdirectoryDropdown.value;
        currentIndex = (currentIndex + 1) % imagesBySubdirectory[selectedSubdirectory].length;
        loadImages(imagesBySubdirectory[selectedSubdirectory]);
      }

      function prevImage() {
        const selectedSubdirectory = subdirectoryDropdown.value;
        currentIndex = (currentIndex - 1 + imagesBySubdirectory[selectedSubdirectory].length) % imagesBySubdirectory[selectedSubdirectory].length;
        loadImages(imagesBySubdirectory[selectedSubdirectory]);
      }

      function togglePlayPause() {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        } else {
          intervalId = setInterval(nextImage, speed);
        }
      }

      function changeSpeed(amount) {
        speed += amount * -100; // Increase or decrease speed by 100 milliseconds
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = setInterval(nextImage, speed);
        }
      }

      // Load initial images based on the selected subdirectory
      const initialSubdirectory = subdirectoryDropdown.value;
      loadImages(imagesBySubdirectory[initialSubdirectory]);
    </script>

    </body>
    </html>
  `;

  fs.writeFileSync('anim.html', htmlContent);
}

const imagesBySubdirectory = getImagesInDirectory(currentDirectory);
generateHTML(imagesBySubdirectory);
console.log('HTML file generated successfully.');