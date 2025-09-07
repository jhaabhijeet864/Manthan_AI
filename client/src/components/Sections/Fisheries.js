import React, { useEffect, useRef } from 'react';
import { 
  SchoolOfFish, 
  FloatingDataIcon, 
  BlinkingDot 
} from '../AnimatedElements/AnimatedElements';
import useChartRenderer from '../../hooks/useChartRenderer';

const Fisheries = ({ isActive }) => {
  const { renderChart } = useChartRenderer();
  const catchChartRef = useRef(null);
  const effortChartRef = useRef(null);

  useEffect(() => {
    if (isActive) {
      // Render charts when section becomes active
      setTimeout(() => {
        if (catchChartRef.current) {
          renderChart(catchChartRef.current, 'catchChart');
        }
        if (effortChartRef.current) {
          renderChart(effortChartRef.current, 'effortChart');
        }
      }, 100); // Small delay to ensure DOM is ready
    }
  }, [isActive, renderChart]);

  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <SchoolOfFish />
      <div className="section-header">
        <h2><i className="fas fa-fish"></i> Fisheries Management</h2>
        <div className="section-indicator"></div>
      </div>
      
      <div className="fisheries-grid">
        <div className="chart-panel">
          <FloatingDataIcon icon="fas fa-chart-pie" />
          <div className="panel-header">
            <h3><i className="fas fa-chart-pie"></i> Catch Composition</h3>
          </div>
          <canvas 
            ref={catchChartRef}
            className="fisheries-chart" 
            id="catchChart"
          ></canvas>
        </div>
        
        <div className="chart-panel">
          <FloatingDataIcon icon="fas fa-anchor" delay={1} />
          <div className="panel-header">
            <h3><i className="fas fa-chart-bar"></i> Effort vs Yield</h3>
          </div>
          <canvas 
            ref={effortChartRef}
            className="fisheries-chart" 
            id="effortChart"
          ></canvas>
        </div>
        
        <div className="compliance-panel">
          <FloatingDataIcon icon="fas fa-shield-alt" delay={2} />
          <div className="panel-header">
            <h3><i className="fas fa-shield-alt"></i> Compliance & Alerts</h3>
          </div>
          <div className="alert-list">
            <div className="alert-item high">
              <BlinkingDot />
              <div className="alert-icon"><i className="fas fa-exclamation-triangle"></i></div>
              <div className="alert-content">
                <div className="alert-title">Bycatch Alert</div>
                <div className="alert-location">Arabian Sea Zone 3</div>
                <div className="alert-time">2 hours ago</div>
              </div>
            </div>
            <div className="alert-item medium">
              <BlinkingDot delay={1} />
              <div className="alert-icon"><i className="fas fa-algae"></i></div>
              <div className="alert-content">
                <div className="alert-title">HAB Risk Warning</div>
                <div className="alert-location">Bay of Bengal</div>
                <div className="alert-time">4 hours ago</div>
              </div>
            </div>
            <div className="alert-item low">
              <div className="alert-icon"><i className="fas fa-info-circle"></i></div>
              <div className="alert-content">
                <div className="alert-title">Quota Update</div>
                <div className="alert-location">West Coast Fisheries</div>
                <div className="alert-time">6 hours ago</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Fisheries;
