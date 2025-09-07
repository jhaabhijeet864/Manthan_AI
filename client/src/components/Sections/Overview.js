import React, { useEffect, useRef } from 'react';
import { 
  FloatingIconMini, 
  FloatingMarineIcon, 
  FloatingDataIcon, 
  DataStreamLine, 
  PulseDot 
} from '../AnimatedElements/AnimatedElements';

const Overview = ({ isActive }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (chartRef.current && isActive) {
      drawChart(chartRef.current);
    }
  }, [isActive]);

  const drawChart = (canvas) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Generate sample data
    const data = [];
    for (let i = 0; i < 20; i++) {
      data.push({
        x: i,
        value: 25 + Math.sin(i * 0.5) * 3 + Math.random() * 2
      });
    }
    
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding - 30;
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue;
    
    // Draw background
    ctx.fillStyle = 'rgba(0, 255, 255, 0.05)';
    ctx.fillRect(padding, padding + 30, chartWidth, chartHeight);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(72, 52, 212, 0.2)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const y = padding + 30 + (i * chartHeight / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }
    
    // Draw area
    const gradient = ctx.createLinearGradient(0, padding + 30, 0, height - padding);
    gradient.addColorStop(0, 'rgba(72, 52, 212, 0.4)');
    gradient.addColorStop(1, 'rgba(72, 52, 212, 0.05)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    
    data.forEach((point, index) => {
      const x = padding + (index * chartWidth / (data.length - 1));
      const y = padding + 30 + ((maxValue - point.value) / range) * chartHeight;
      ctx.lineTo(x, y);
    });
    
    ctx.lineTo(width - padding, height - padding);
    ctx.closePath();
    ctx.fill();
    
    // Draw line
    ctx.strokeStyle = '#4834d4';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#4834d4';
    ctx.shadowBlur = 8;
    
    ctx.beginPath();
    data.forEach((point, index) => {
      const x = padding + (index * chartWidth / (data.length - 1));
      const y = padding + 30 + ((maxValue - point.value) / range) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  };

  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2><i className="fas fa-chart-line"></i> Ocean Intelligence Overview</h2>
        <div className="section-indicator"></div>
      </div>
      
      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card temperature">
          <FloatingIconMini icon="fas fa-thermometer-half" />
          <div className="kpi-icon"><i className="fas fa-thermometer-half"></i></div>
          <div className="kpi-content">
            <div className="kpi-value">28.7°C</div>
            <div className="kpi-label">Avg SST</div>
            <div className="kpi-trend positive">↑ 0.3°C</div>
          </div>
          <div className="kpi-spark"></div>
        </div>
        
        <div className="kpi-card salinity">
          <FloatingIconMini icon="fas fa-tint" />
          <div className="kpi-icon"><i className="fas fa-tint"></i></div>
          <div className="kpi-content">
            <div className="kpi-value">35.2‰</div>
            <div className="kpi-label">Salinity</div>
            <div className="kpi-trend stable">→ Stable</div>
          </div>
          <div className="kpi-spark"></div>
        </div>
        
        <div className="kpi-card biodiversity">
          <FloatingIconMini icon="fas fa-dna" />
          <div className="kpi-icon"><i className="fas fa-dna"></i></div>
          <div className="kpi-content">
            <div className="kpi-value">2,847</div>
            <div className="kpi-label">eDNA Matches</div>
            <div className="kpi-trend positive">↑ 156</div>
          </div>
          <div className="kpi-spark"></div>
        </div>
        
        <div className="kpi-card compliance">
          <FloatingIconMini icon="fas fa-shield-alt" />
          <div className="kpi-icon"><i className="fas fa-shield-alt"></i></div>
          <div className="kpi-content">
            <div className="kpi-value">94.2%</div>
            <div className="kpi-label">Compliance</div>
            <div className="kpi-trend positive">↑ 2.1%</div>
          </div>
          <div className="kpi-spark"></div>
        </div>
        
        <div className="kpi-card alerts">
          <FloatingIconMini icon="fas fa-exclamation-triangle" />
          <div className="kpi-icon"><i className="fas fa-exclamation-triangle"></i></div>
          <div className="kpi-content">
            <div className="kpi-value">7</div>
            <div className="kpi-label">Active Alerts</div>
            <div className="kpi-trend negative">↑ 2</div>
          </div>
          <div className="kpi-spark"></div>
        </div>
      </div>

      {/* Heatmap and Charts */}
      <div className="overview-grid">
        <div className="heatmap-container">
          <FloatingMarineIcon icon="fas fa-fish" />
          <FloatingMarineIcon icon="fas fa-ship" delay={1} />
          <div className="panel-header">
            <h3><i className="fas fa-map"></i> India EEZ Heat Map</h3>
            <div className="panel-controls">
              <button className="control-btn active">SST</button>
              <button className="control-btn">Salinity</button>
              <button className="control-btn">Biodiversity</button>
            </div>
          </div>
          <div className="heatmap-mock">
            <div className="map-region hot-spot" data-temp="29.2°C" data-location="Arabian Sea"></div>
            <div className="map-region medium-spot" data-temp="27.8°C" data-location="Bay of Bengal"></div>
            <div className="map-region cool-spot" data-temp="26.1°C" data-location="Indian Ocean"></div>
            <div className="map-region warm-spot" data-temp="28.9°C" data-location="Lakshadweep Sea"></div>
            <div className="pulse-indicator" style={{top: '30%', left: '20%'}}></div>
            <div className="pulse-indicator" style={{top: '45%', right: '25%'}}></div>
            <div className="pulse-indicator" style={{bottom: '30%', left: '40%'}}></div>
            <DataStreamLine style={{top: '25%', left: '15%', width: '60%'}} />
            <DataStreamLine style={{top: '65%', left: '25%', width: '50%'}} delay={2} />
          </div>
        </div>
        
        <div className="charts-container">
          <div className="chart-panel">
            <FloatingDataIcon icon="fas fa-chart-line" />
            <div className="panel-header">
              <h3><i className="fas fa-chart-area"></i> SST Trends (7 Days)</h3>
            </div>
            <canvas ref={chartRef} className="trend-chart" id="sstChart"></canvas>
          </div>
          
          <div className="chart-panel">
            <FloatingDataIcon icon="fas fa-seedling" delay={1} />
            <div className="panel-header">
              <h3><i className="fas fa-leaf"></i> Biodiversity Hotspots</h3>
            </div>
            <div className="hotspot-list">
              <div className="hotspot-item">
                <PulseDot />
                <div className="hotspot-location">Andaman Sea</div>
                <div className="hotspot-value">1,247 species</div>
                <div className="hotspot-bar" style={{width: '95%'}}></div>
              </div>
              <div className="hotspot-item">
                <PulseDot delay={1} />
                <div className="hotspot-location">Lakshadweep</div>
                <div className="hotspot-value">892 species</div>
                <div className="hotspot-bar" style={{width: '78%'}}></div>
              </div>
              <div className="hotspot-item">
                <PulseDot delay={2} />
                <div className="hotspot-location">Gulf of Mannar</div>
                <div className="hotspot-value">634 species</div>
                <div className="hotspot-bar" style={{width: '65%'}}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Overview;
