// Tab Logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
        
        if (btn.dataset.target === 'tab-dashboard' && !window.dashboardLoaded) {
            loadDashboard();
        }
    });
});

let currentPredictionData = null; // Store for PDF export

// Single Prediction
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    const resultContainer = document.getElementById('result-container');
    const submitBtn = document.querySelector('.primary-btn');

    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    submitBtn.disabled = true;
    resultContainer.classList.add('hidden');

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    data.senior_citizen = parseInt(data.senior_citizen);
    data.tenure = parseInt(data.tenure);
    data.monthly_charges = parseFloat(data.monthly_charges);
    data.total_charges = parseFloat(data.total_charges);
    data.num_services = parseInt(data.num_services);

    currentPredictionData = data; // Save for PDF

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Prediction failed');
        const result = await response.json();
        
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        submitBtn.disabled = false;
        resultContainer.classList.remove('hidden');
        
        // Status & Probability
        const statusEl = document.getElementById('result-status');
        if (result.churn_prediction === 1) {
            statusEl.textContent = 'High Risk of Churn';
            statusEl.className = 'status-badge status-churn';
        } else {
            statusEl.textContent = 'Likely to Retain';
            statusEl.className = 'status-badge status-retain';
        }
        
        const probPct = (result.churn_probability * 100).toFixed(1);
        document.getElementById('prob-text').textContent = `${probPct}%`;
        setTimeout(() => { document.getElementById('progress-bar').style.width = `${probPct}%`; }, 100);

        // Render SHAP
        const shapContainer = document.getElementById('shap-bars');
        shapContainer.innerHTML = '';
        if (result.shap_values && Object.keys(result.shap_values).length > 0) {
            const maxShap = Math.max(...Object.values(result.shap_values).map(Math.abs));
            for (const [feat, val] of Object.entries(result.shap_values)) {
                if (Math.abs(val) < 0.01) continue; // Skip near-zero impact
                const pct = (Math.abs(val) / maxShap) * 100;
                const isPositive = val > 0; // Contributes to churn
                
                const row = document.createElement('div');
                row.className = 'shap-row';
                
                let barHtml = '';
                if (isPositive) {
                    barHtml = `<div class="shap-bar" style="right: 50%; width: 0; background: var(--danger);"></div>
                               <div class="shap-bar" style="left: 50%; width: ${pct/2}%; background: var(--danger);"></div>`;
                } else {
                    barHtml = `<div class="shap-bar" style="left: 50%; width: 0; background: var(--success);"></div>
                               <div class="shap-bar" style="right: 50%; width: ${pct/2}%; background: var(--success);"></div>`;
                }

                row.innerHTML = `
                    <div class="shap-label">${feat.replace('_', ' ')}</div>
                    <div class="shap-bar-container">${barHtml}</div>
                    <div class="shap-val" style="color: ${isPositive ? 'var(--danger)' : 'var(--success)'}">${val > 0 ? '+' : ''}${val.toFixed(2)}</div>
                `;
                shapContainer.appendChild(row);
            }
        } else {
            shapContainer.innerHTML = '<p style="color: #666; font-size: 0.9rem;">SHAP values not available.</p>';
        }

        // Render Recommendations
        const recsList = document.getElementById('recs-list');
        recsList.innerHTML = '';
        result.recommendations.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            recsList.appendChild(li);
        });

    } catch (error) {
        console.error(error);
        alert('An error occurred. Check backend logs.');
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        submitBtn.disabled = false;
    }
});

// PDF Export
document.getElementById('export-pdf-btn').addEventListener('click', async () => {
    if (!currentPredictionData) return;
    try {
        const response = await fetch('/api/export_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPredictionData)
        });
        if (!response.ok) throw new Error('PDF Generation failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'churn_report.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
    } catch (e) {
        alert('Failed to export PDF.');
    }
});

// Bulk Upload
document.getElementById('bulk-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const btnText = document.getElementById('bulk-btn-text');
    const loader = document.getElementById('bulk-btn-loader');
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/predict_bulk', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Bulk process failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
    } catch (e) {
        alert('Failed to process CSV.');
    } finally {
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
});

// File input UI update
document.getElementById('csv-file').addEventListener('change', (e) => {
    const p = document.querySelector('#upload-zone p');
    if (e.target.files.length > 0) {
        p.textContent = `Selected: ${e.target.files[0].name}`;
    } else {
        p.innerHTML = `Drag & Drop your CSV here or <span>click to browse</span>`;
    }
});

// Dashboard
async function loadDashboard() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) return;
        const stats = await res.json();
        
        document.getElementById('stat-total').textContent = stats.total_customers.toLocaleString();
        document.getElementById('stat-rate').textContent = `${stats.churn_rate}%`;

        // Colors
        const colors = ['rgba(79, 70, 229, 0.7)', 'rgba(236, 72, 153, 0.7)', 'rgba(59, 130, 246, 0.7)'];

        // Contract Chart
        const ctxContract = document.getElementById('contractChart').getContext('2d');
        new Chart(ctxContract, {
            type: 'doughnut',
            data: {
                labels: Object.keys(stats.contract_distribution),
                datasets: [{
                    data: Object.values(stats.contract_distribution),
                    backgroundColor: colors,
                    borderColor: '#141414',
                    borderWidth: 2
                }]
            },
            options: { plugins: { title: { display: true, text: 'Contract Types', color: '#fff' }, legend: { labels: { color: '#ccc' } } } }
        });

        // Charges Chart
        const ctxCharges = document.getElementById('chargesChart').getContext('2d');
        new Chart(ctxCharges, {
            type: 'bar',
            data: {
                labels: Object.keys(stats.avg_monthly_charges),
                datasets: [{
                    label: 'Avg Monthly Charge ($)',
                    data: Object.values(stats.avg_monthly_charges),
                    backgroundColor: ['rgba(59, 226, 130, 0.7)', 'rgba(255, 74, 74, 0.7)'],
                    borderRadius: 4
                }]
            },
            options: {
                plugins: { title: { display: true, text: 'Charges by Churn Status', color: '#fff' }, legend: { display: false } },
                scales: {
                    y: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { ticks: { color: '#ccc' }, grid: { display: false } }
                }
            }
        });

        window.dashboardLoaded = true;
    } catch (e) {
        console.error("Failed to load dashboard", e);
    }
}
