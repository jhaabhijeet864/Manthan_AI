import React, { useEffect, useRef } from 'react';
import useChartRenderer from '../../hooks/useChartRenderer';

const Oceanography = ({ isActive }) => {
  const { renderChart } = useChartRenderer();
  const sstChartRef = useRef(null);
  const salinityChartRef = useRef(null);

  useEffect(() => {
    if (isActive) {
      // Render charts when section becomes active
      setTimeout(() => {
        if (sstChartRef.current) {
          renderChart(sstChartRef.current, 'sstTrendChart');
        }
        if (salinityChartRef.current) {
          renderChart(salinityChartRef.current, 'salinityChart');
        }
      }, 100); // Small delay to ensure DOM is ready
    }
  }, [isActive, renderChart]);

  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2><i className="fas fa-thermometer-half"></i> Ocean Data Analysis</h2>
        <div className="section-indicator"></div>
      </div>
      
      <div className="oceanography-grid">
        <div className="chart-panel large">
          <div className="panel-header">
            <h3><i className="fas fa-thermometer-half"></i> Sea Surface Temperature</h3>
            <div className="time-selector">
              <button className="time-btn active">24H</button>
              <button className="time-btn">7D</button>
              <button className="time-btn">30D</button>
            </div>
          </div>
          <canvas 
            ref={sstChartRef}
            className="ocean-chart" 
            id="sstTrendChart"
          ></canvas>
        </div>
        
        <div className="chart-panel large">
          <div className="panel-header">
            <h3><i className="fas fa-tint"></i> Salinity Levels</h3>
            <div className="time-selector">
              <button className="time-btn active">24H</button>
              <button className="time-btn">7D</button>
              <button className="time-btn">30D</button>
            </div>
          </div>
          <canvas 
            ref={salinityChartRef}
            className="ocean-chart" 
            id="salinityChart"
          ></canvas>
        </div>
        
        <div className="metadata-panel">
          <div className="panel-header">
            <h3><i className="fas fa-satellite"></i> Data Sources</h3>
          </div>
          <div className="metadata-list">
            <div className="metadata-item">
              <div className="metadata-icon"><i className="fas fa-satellite-dish"></i></div>
              <div>
                <div className="metadata-title">MODIS Aqua</div>
                <div className="metadata-desc">Last update: 12:34 UTC</div>
              </div>
            </div>
            <div className="metadata-item">
              <div className="metadata-icon"><i className="fas fa-ship"></i></div>
              <div>
                <div className="metadata-title">In-Situ Buoys</div>
                <div className="metadata-desc">42 active stations</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Oceanography;
