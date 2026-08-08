const uploadForm = document.getElementById('upload-form');
const messageBox = document.getElementById('form-message');
const summarySection = document.getElementById('summary');
const vizSection = document.getElementById('visualizations');
const prescriptionsSection = document.getElementById('prescriptions');
const tableSection = document.getElementById('table');
const prescriptionGrid = document.getElementById('prescription-grid');
const resultTableBody = document.querySelector('#result-table tbody');

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideSections();
    showMessage('Analyzing dataset, please wait...', 'success');

    const files = document.getElementById('data-files').files;
    if (!files.length) {
        showMessage('Please select at least one CSV file before continuing.', 'error');
        return;
    }

    const data = new FormData();
    Array.from(files).forEach((file) => data.append('data_files', file));
    try {
        const response = await fetch(analyzeEndpoint, {
            method: 'POST',
            body: data,
        });
        const payload = await response.json();

        if (!payload.success) {
            showMessage(payload.message || 'Analysis failed. Please try a different dataset.', 'error');
            return;
        }

        showMessage('Analysis complete. Review the dashboard below.', 'success');
        renderResults(payload.result);
    } catch (error) {
        console.error(error);
        showMessage('An unexpected error occurred while analyzing data.', 'error');
    }
});

function showMessage(text, type) {
    messageBox.className = `message message--${type}`;
    messageBox.textContent = text;
    messageBox.style.display = 'block';
}

function hideSections() {
    [summarySection, vizSection, prescriptionsSection, tableSection].forEach((el) => {
        if (el) el.classList.add('card--hidden');
    });
    messageBox.style.display = 'none';
}

function renderResults(result) {
    renderSummary(result.summary);
    renderCharts(result.charts, result.summary);
    renderPrescriptions(result.prescriptions);
    renderTable(result.daily_preview);
    [summarySection, vizSection, prescriptionsSection, tableSection].forEach((el) => {
        el.classList.remove('card--hidden');
    });
}

function renderSummary(summary) {
    document.getElementById('metric-bci').textContent = summary.bci_mean.toFixed(3);
    document.getElementById('metric-state').textContent = summary.last_state;
    document.getElementById('metric-sleep').textContent = summary.sleep_average ? `${summary.sleep_average.toFixed(1)} hrs` : 'N/A';
    document.getElementById('hero-bci').textContent = summary.bci_mean.toFixed(3);
}

function renderCharts(charts, summary) {
    const today = new Date();
    const stateProbabilities = summary.forecast.state_probability;

    Plotly.newPlot('chart-bci', [{
        x: charts.dates,
        y: charts.bci,
        mode: 'lines+markers',
        marker: { color: '#34d1ff' },
        fill: 'tozeroy',
        name: 'BCI'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 24, b: 48 },
        xaxis: { tickangle: -45 },
        yaxis: { title: 'BCI', range: [0, 1] }
    });

    Plotly.newPlot('chart-entropy', [{
        x: charts.dates,
        y: charts.entropy,
        type: 'bar',
        marker: { color: '#81c2ff' },
        name: 'Entropy'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 24, b: 48 },
        xaxis: { tickangle: -45 },
        yaxis: { title: 'Entropy' }
    });

    Plotly.newPlot('chart-anomaly', [{
        x: charts.dates,
        y: charts.anomaly,
        type: 'bar',
        marker: { color: '#ff9f75' },
        name: 'Anomaly score'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 24, b: 48 },
        xaxis: { tickangle: -45 },
        yaxis: { title: 'Anomaly score', range: [0, 1] }
    });

    Plotly.newPlot('chart-sleep', [
        {
            x: charts.dates,
            y: charts.sleep,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#b97bff' },
            name: 'Sleep hours'
        },
        {
            x: charts.dates,
            y: charts.fatigue,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#ff5d9f' },
            name: 'Fatigue score'
        }
    ], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 24, b: 48 },
        xaxis: { tickangle: -45 },
        yaxis: { title: 'Sleep / fatigue' }
    });

    const forecastStates = Object.keys(stateProbabilities);
    const forecastValues = Object.values(stateProbabilities);

    Plotly.newPlot('chart-forecast', [{
        values: forecastValues,
        labels: forecastStates,
        type: 'pie',
        hole: 0.54,
        marker: { colors: ['#34d1ff', '#ffba60', '#ff6f6f'] },
        textinfo: 'label+percent'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 20, b: 20 }
    });
}

function renderPrescriptions(cards) {
    prescriptionGrid.innerHTML = '';
    cards.forEach((card) => {
        const item = document.createElement('div');
        item.className = `prescription-card prescription-card--${card.type}`;
        item.innerHTML = `<h4>${card.title}</h4><p>${card.description}</p>`;
        prescriptionGrid.appendChild(item);
    });
}

function renderTable(rows) {
    resultTableBody.innerHTML = '';
    rows.forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.Date}</td>
            <td>${Number(row.BCI).toFixed(3)}</td>
            <td>${Number(row.Entropy).toFixed(3)}</td>
            <td>${Number(row.AnomalyScore).toFixed(3)}</td>
            <td>${row.State}</td>
            <td>${row.SleepHours ? Number(row.SleepHours).toFixed(1) : '—'}</td>
            <td>${Number(row.FatigueScore).toFixed(3)}</td>
        `;
        resultTableBody.appendChild(tr);
    });
}
