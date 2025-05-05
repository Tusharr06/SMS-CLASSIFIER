document.addEventListener("DOMContentLoaded", function () {
  // Tab navigation
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabId = button.getAttribute("data-tab");

      // Update active button
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");

      // Show active tab
      tabPanes.forEach((pane) => {
        pane.classList.remove("active");
        if (pane.id === tabId) {
          pane.classList.add("active");
        }
      });
    });
  });

  // Result tabs navigation
  const resultTabButtons = document.querySelectorAll(".result-tab-btn");
  const resultTabPanes = document.querySelectorAll(".result-tab-pane");

  resultTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabId = button.getAttribute("data-result-tab");

      // Update active button
      resultTabButtons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");

      // Show active tab
      resultTabPanes.forEach((pane) => {
        pane.classList.remove("active");
        if (pane.id === tabId) {
          pane.classList.add("active");
        }
      });
    });
  });

  // Demo button
  const demoButton = document.getElementById("demo-button");
  demoButton.addEventListener("click", () => {
    fetch("/api/demo-text")
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("text-input").value = data.text;
      })
      .catch((error) => console.error("Error loading demo text:", error));
  });

  // Clear button
  const clearButton = document.getElementById("clear-btn");
  clearButton.addEventListener("click", () => {
    document.getElementById("text-input").value = "";
    document.getElementById("results-container").style.display = "none";
  });

  // Analyze button
  const analyzeButton = document.getElementById("analyze-btn");
  analyzeButton.addEventListener("click", analyzeText);

  // Analyze text function
  function analyzeText() {
    const textInput = document.getElementById("text-input").value.trim();

    if (!textInput) {
      alert("Please enter some text to analyze.");
      return;
    }

    // Show loader
    const loader = document.getElementById("loader");
    loader.style.display = "block";

    // Hide results
    document.getElementById("results-container").style.display = "none";

    // Update progress bar animation
    const progressFill = document.querySelector(".progress-fill");
    const statusText = document.getElementById("status-text");

    progressFill.style.width = "25%";
    statusText.textContent = "Analyzing text...";

    setTimeout(() => {
      progressFill.style.width = "50%";
      statusText.textContent = "Scanning for URLs...";
    }, 500);

    setTimeout(() => {
      progressFill.style.width = "75%";
      statusText.textContent = "Analyzing website safety...";
    }, 1000);

    // Send API request
    fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: textInput }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        // Complete progress bar
        progressFill.style.width = "100%";
        statusText.textContent = "Analysis complete!";

        setTimeout(() => {
          // Hide loader
          loader.style.display = "none";

          // Show results
          displayResults(data);
          document.getElementById("results-container").style.display = "block";
        }, 500);
      })
      .catch((error) => {
        console.error("Error:", error);
        loader.style.display = "none";
        alert("An error occurred during analysis. Please try again.");
      });
  }

  // Display results function
  function displayResults(data) {
    // Spam analysis results
    const predictionBox = document.getElementById("prediction-box");

    let mainProb, otherProb, mainLabel, otherLabel, mainColor, otherColor;
    if (data.prediction === "Spam") {
      mainProb = data.spam_probability;
      otherProb = data.not_spam_probability;
      mainLabel = "Spam";
      otherLabel = "Not Spam";
      mainColor = "#ff7675";
      otherColor = "#00b894";
    } else {
      mainProb = data.not_spam_probability;
      otherProb = data.spam_probability;
      mainLabel = "Not Spam";
      otherLabel = "Spam";
      mainColor = "#00b894";
      otherColor = "#ff7675";
    }

    predictionBox.style.backgroundColor = mainColor;
    predictionBox.innerHTML = `
            <h2>${mainLabel}</h2>
            <p>Confidence: ${data.confidence.toFixed(1)}%</p>
        `;

    // Always show 'Spam Probability' and spam probability value in the gauge
    const gaugeLabel = document.getElementById("gauge-label");
    if (gaugeLabel) {
      gaugeLabel.textContent = "Spam Probability";
    }
    const spamGaugeColor = data.spam_probability > 50 ? "#ff7675" : "#00b894";
    createGaugeChart("spam-gauge", data.spam_probability, spamGaugeColor);

    // Pie chart
    createPieChart(
      "classification-pie",
      [mainLabel, otherLabel],
      [mainProb, otherProb],
      [mainColor, otherColor]
    );

    // URL analysis results
    const urlResults = document.getElementById("url-results");
    const noUrlsMessage = document.getElementById("no-urls-message");
    const urlChartContainer = document.getElementById("url-chart-container");
    const urlCardsContainer = document.getElementById("url-cards-container");
    const overallAssessment = document.getElementById("overall-assessment");

    // Clear previous URL results
    urlCardsContainer.innerHTML = "";

    if (data.urls && data.urls.length > 0) {
      noUrlsMessage.style.display = "none";
      urlChartContainer.style.display = "block";

      // Create URL bar chart
      createURLBarChart("url-chart", data.urls);

      // Create URL cards
      data.urls.forEach((url, index) => {
        const urlCard = createURLCard(url, index);
        urlCardsContainer.appendChild(urlCard);
      });

      // Calculate average trust score
      const avgScore =
        data.urls.reduce((sum, url) => sum + url.trust_score, 0) /
        data.urls.length;

      // Set overall assessment
      let assessmentColor, assessmentText;
      if (avgScore >= 70) {
        assessmentColor = "#00b894";
        assessmentText = "Links appear trustworthy";
      } else if (avgScore >= 50) {
        assessmentColor = "#74b9ff";
        assessmentText = "Links have moderate trust";
      } else {
        assessmentColor = "#ff7675";
        assessmentText = "Links appear suspicious";
      }

      overallAssessment.style.display = "block";
      overallAssessment.innerHTML = `
                <div class="assessment-box" style="background-color: ${assessmentColor};">
                    <h3>Overall URL Assessment</h3>
                    <p>Average trust score: ${avgScore.toFixed(
                      1
                    )}/100 - ${assessmentText}</p>
                </div>
            `;
    } else {
      noUrlsMessage.style.display = "block";
      urlChartContainer.style.display = "none";
      overallAssessment.style.display = "none";
    }
  }

  // Create URL card
  function createURLCard(urlInfo, index) {
    const card = document.createElement("div");
    card.className = "url-card";

    const title = document.createElement("h4");
    title.className = "url-title";
    title.textContent = `URL ${index + 1}: ${urlInfo.domain}`;

    const content = document.createElement("div");
    content.className = "url-card-content";

    // URL gauge chart container
    const gaugeContainer = document.createElement("div");
    gaugeContainer.className = "url-gauge";
    const gaugeCanvas = document.createElement("canvas");
    gaugeCanvas.id = `url-gauge-${index}`;
    gaugeContainer.appendChild(gaugeCanvas);

    // URL details
    const details = document.createElement("div");
    details.className = "url-details";

    details.innerHTML = `
            <p><strong>Full URL:</strong> ${urlInfo.url}</p>
            <p><strong>Classification:</strong> ${urlInfo.classification}</p>
        `;

    if (urlInfo.risk_factors && urlInfo.risk_factors.length > 0) {
      const riskFactors = document.createElement("div");
      riskFactors.innerHTML = "<p><strong>Risk Factors:</strong></p>";
      const riskList = document.createElement("ul");
      urlInfo.risk_factors.forEach((factor) => {
        const item = document.createElement("li");
        item.textContent = factor;
        riskList.appendChild(item);
      });
      riskFactors.appendChild(riskList);
      details.appendChild(riskFactors);
    }

    if (urlInfo.security_features && urlInfo.security_features.length > 0) {
      const securityFeatures = document.createElement("div");
      securityFeatures.innerHTML = "<p><strong>Security Features:</strong></p>";
      const featureList = document.createElement("ul");
      urlInfo.security_features.forEach((feature) => {
        const item = document.createElement("li");
        item.textContent = feature;
        featureList.appendChild(item);
      });
      securityFeatures.appendChild(featureList);
      details.appendChild(securityFeatures);
    }

    content.appendChild(gaugeContainer);
    content.appendChild(details);

    card.appendChild(title);
    card.appendChild(content);

    // Create gauge chart for URL trust score
    setTimeout(() => {
      createGaugeChart(
        `url-gauge-${index}`,
        urlInfo.trust_score,
        urlInfo.trust_score >= 50 ? "#00b894" : "#ff7675"
      );
    }, 0);

    return card;
  }

  // Create gauge chart
  function createGaugeChart(canvasId, value, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart if it exists
    if (window[canvasId + "Chart"]) {
      window[canvasId + "Chart"].destroy();
    }

    // Remove any existing center text
    const parent = canvas.parentNode;
    const oldCenterText = parent.querySelector(".gauge-center-text");
    if (oldCenterText) parent.removeChild(oldCenterText);

    const ctx = canvas.getContext("2d");
    window[canvasId + "Chart"] = new Chart(ctx, {
      type: "doughnut",
      data: {
        datasets: [
          {
            data: [value, 100 - value],
            backgroundColor: [color, "#f1f2f6"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        circumference: 180,
        rotation: -90,
        cutout: "80%",
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        animation: {
          animateRotate: true,
          animateScale: true,
        },
      },
    });

    // Only add center text for gauge containers
    if (
      parent.classList.contains("gauge-chart-container") ||
      parent.classList.contains("url-gauge")
    ) {
      const centerText = document.createElement("div");
      centerText.className = "gauge-center-text";
      centerText.innerHTML = `${value.toFixed(1)}%`;
      parent.appendChild(centerText);
    }
  }

  // Create pie chart
  function createPieChart(canvasId, labels, data, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart if it exists
    if (window[canvasId + "Chart"]) {
      window[canvasId + "Chart"].destroy();
    }

    const ctx = canvas.getContext("2d");
    window[canvasId + "Chart"] = new Chart(ctx, {
      type: "pie",
      data: {
        labels: labels,
        datasets: [
          {
            data: data,
            backgroundColor: colors,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              padding: 20,
              font: {
                size: 14,
              },
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `${context.label}: ${context.raw.toFixed(1)}%`;
              },
            },
          },
        },
        animation: {
          animateRotate: true,
          animateScale: true,
        },
      },
    });
  }

  // Create URL bar chart
  function createURLBarChart(canvasId, urls) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart if it exists
    if (window[canvasId + "Chart"]) {
      window[canvasId + "Chart"].destroy();
    }

    const ctx = canvas.getContext("2d");
    const labels = urls.map((url, index) => `URL ${index + 1}`);
    const data = urls.map((url) => url.trust_score);
    const backgroundColors = data.map((score) =>
      score >= 80
        ? "#00b894"
        : score >= 60
        ? "#74b9ff"
        : score >= 40
        ? "#fdcb6e"
        : "#ff7675"
    );

    window[canvasId + "Chart"] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            data: data,
            backgroundColor: backgroundColors,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `Trust Score: ${context.raw.toFixed(1)}`;
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: "Trust Score",
            },
          },
        },
        animation: {
          duration: 1000,
        },
      },
    });

    // Failsafe: Remove any .gauge-center-text from the bar chart container
    const barChartContainer = canvas.parentNode;
    const strayTexts = barChartContainer.querySelectorAll(".gauge-center-text");
    strayTexts.forEach((el) => el.remove());
  }

  // Create indicators chart for "How It Works" tab
  const spamIndicators = {
    "Urgency Language": 85,
    "Suspicious Links": 78,
    "Request for Information": 72,
    "Grammar Errors": 65,
    "Money Offers": 92,
    "Account Warnings": 81,
  };

  const indicatorsCanvas = document.getElementById("indicators-chart");
  new Chart(indicatorsCanvas, {
    type: "bar",
    data: {
      labels: Object.keys(spamIndicators),
      datasets: [
        {
          label: "Frequency (%)",
          data: Object.values(spamIndicators),
          backgroundColor: "#38b6ff",
          borderWidth: 0,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: "#444444",
          },
          ticks: {
            color: "white",
          },
          title: {
            display: true,
            text: "Frequency in Spam Messages (%)",
            color: "white",
          },
        },
        y: {
          grid: {
            color: "#444444",
          },
          ticks: {
            color: "white",
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: "Common Indicators in Spam Messages",
          color: "white",
          font: {
            size: 16,
          },
        },
      },
    },
  });
});
