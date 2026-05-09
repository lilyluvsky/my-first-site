function toggleDarkMode() {
  document.body.classList.toggle("dark");

  if (document.body.classList.contains("dark")) {
    localStorage.setItem("darkMode", "enabled");
  } else {
    localStorage.setItem("darkMode", "disabled");
  }
}

function loadDarkMode() {
  if (localStorage.getItem("darkMode") === "enabled") {
    document.body.classList.add("dark");
  }
}

function copyProjectSummary() {
  const summary = "Quant Research Lab: two connected projects covering A-share factor research with Jupyter Notebook, CSV files and reports, plus a Python moving-average backtesting engine with metrics, CSV results and an equity curve.";
  navigator.clipboard.writeText(summary).then(() => {
    alert("Project summary copied.");
  }).catch(() => {
    alert(summary);
  });
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

function handleBackToTopButton() {
  const button = document.getElementById("backToTop");

  if (!button) {
    return;
  }

  button.style.display = window.scrollY > 300 ? "block" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
  loadDarkMode();
  handleBackToTopButton();
});

window.addEventListener("scroll", handleBackToTopButton);
