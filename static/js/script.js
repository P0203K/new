const alphabetBtn = document.getElementById('alphabetBtn');
const numbersBtn = document.getElementById('numbersBtn');
const gridContainer = document.getElementById('gridContainer');
// const imageGallery = document.getElementById('imageGallery');
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modalImg');
const closeBtn = document.getElementsByClassName('close')[0];

const alphabetImages = [
  { path: "static/alphabet/A/A.jpg", name: "A" },
  { path: "static/alphabet/B/B.jpg", name: "B" },
  { path: "static/alphabet/C/C.jpg", name: "C" },
  { path: "static/alphabet/D/D.jpg", name: "D" },
  { path: "static/alphabet/E/E.jpg", name: "E" },
  { path: "static/alphabet/F/F.jpg", name: "F" },
  { path: "static/alphabet/G/G.jpg", name: "G" },
  { path: "static/alphabet/H/H.jpg", name: "H" },
  { path: "static/alphabet/I/I.jpg", name: "I" },
  { path: "static/alphabet/J/J.jpg", name: "J" },
  { path: "static/alphabet/K/K.jpg", name: "K" },
  { path: "static/alphabet/L/L.jpg", name: "L" },
  { path: "static/alphabet/M/M.jpg", name: "M" },
  { path: "static/alphabet/N/N.jpg", name: "N" },
  { path: "static/alphabet/O/O.jpg", name: "O" },
  { path: "static/alphabet/P/P.jpg", name: "P" },
  { path: "static/alphabet/Q/Q.jpg", name: "Q" },
  { path: "static/alphabet/R/R.jpg", name: "R" },
  { path: "static/alphabet/S/S.jpg", name: "S" },
  { path: "static/alphabet/T/T.jpg", name: "T" },
  { path: "static/alphabet/U/U.jpg", name: "U" },
  { path: "static/alphabet/V/V.jpg", name: "V" },
  { path: "static/alphabet/W/W.jpg", name: "W" },
  { path: "static/alphabet/X/X.jpg", name: "X" },
  { path: "static/alphabet/Y/Y.jpg", name: "Y" },
  { path: "static/alphabet/Z/Z.jpg", name: "Z" },
];

const numberImages = [
  { path: "static/number/1/1.jpg", name: "1" },
  { path: "static/number/2/1.jpg", name: "2" },
  { path: "static/number/3/1.jpg", name: "3" },
  { path: "static/number/4/1.jpg", name: "4" },
  { path: "static/number/5/2.jpg", name: "5" },
  { path: "static/number/6/1.jpg", name: "6" },
  { path: "static/number/7/1.jpg", name: "7" },
  { path: "static/number/8/1.jpg", name: "8" },
  { path: "static/number/9/1.jpg", name: "9" },
];

alphabetBtn.addEventListener('click', () => {
    displayImages(alphabetImages);
  });
  
  numbersBtn.addEventListener('click', () => {
    displayImages(numberImages);
  });
  
  function displayImages(imagesArray) {
    gridContainer.innerHTML = ''; // Clear previous images
    imagesArray.forEach(item => {
      const gridItem = document.createElement('div');
      gridItem.className = 'grid-item';
  
      const img = document.createElement('img');
      img.src = item.path;
      img.alt = item.name;
      gridItem.appendChild(img);
  
      const text = document.createElement('div');
      text.textContent = item.name;
      text.className = 'image-name';
      gridItem.appendChild(text);
  
      gridContainer.appendChild(gridItem);
  
      // Add click event listener to open modal
      gridItem.addEventListener('click', () => {
        openModal(item.path);
      });
    });
  }
  
  function openModal(imgPath) {
    modal.style.display = 'block';
    modalImg.src = imgPath;
  }
  
  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
  
  window.addEventListener('click', (event) => {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  });

// Display images in the gallery
// images.forEach(image => {
//     const imgElement = document.createElement('img');
//     imgElement.src = image.path;
//     imgElement.alt = image.name;
//     imgElement.addEventListener('click', () => openModal(image));
//     imageGallery.appendChild(imgElement);
// });

// // Open modal with image details
// function openModal(image) {
//     modal.style.display = 'block';
//     modalContent.innerHTML = `
//         <img src="${image.path}" alt="${image.name}">
//         <div>${image.name}</div>
//     `;
// }

// // Close modal
// closeBtn.addEventListener('click', () => {
//     modal.style.display = 'none';
// });

// window.addEventListener('click', (event) => {
//     if (event.target === modal) {
//         modal.style.display = 'none';
//     }
// });