window.addEventListener("load", () => {
    const title = document.querySelector(".cl-title, .cl__title");
    if (!title) return;
  
    // Replace text
    title.textContent = "VoxelInsight";
  
    // Prepend logo
    const logo = document.createElement("img");
    logo.src = "/public/VoxelInsightLogo.png";
    logo.alt = "VoxelInsight";
    logo.style.height = "28px";
    logo.style.marginRight = "8px";
    title.prepend(logo);
  });
  