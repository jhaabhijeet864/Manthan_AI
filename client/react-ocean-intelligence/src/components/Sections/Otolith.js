import React from 'react';

const Otolith = ({ isActive }) => {
  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2><i className="fas fa-microscope"></i> Otolith Shape Analysis</h2>
        <div className="section-indicator"></div>
      </div>
      
      <div className="otolith-grid">
        <div className="shape-panel">
          <div className="panel-header">
            <h3><i className="fas fa-search"></i> Shape Recognition</h3>
          </div>
          <div className="otolith-viewer">
            <div className="otolith-shape">
              <div className="shape-outline"></div>
              <div className="measurement-points">
                <div className="point" style={{top: '20%', left: '30%'}}></div>
                <div className="point" style={{top: '40%', right: '25%'}}></div>
                <div className="point" style={{bottom: '25%', left: '45%'}}></div>
              </div>
            </div>
            <div className="shape-controls">
              <button className="shape-btn active">Analyze</button>
              <button className="shape-btn">Compare</button>
              <button className="shape-btn">Export</button>
            </div>
          </div>
        </div>
        
        <div className="morphometric-panel">
          <div className="panel-header">
            <h3><i className="fas fa-ruler"></i> Morphometric Analysis</h3>
          </div>
          <div className="descriptors-list">
            <div className="descriptor-item">
              <div className="descriptor-label">Length (mm)</div>
              <div className="descriptor-value">12.4</div>
              <div className="descriptor-bar">
                <div className="bar-fill" style={{width: '82%'}}></div>
              </div>
            </div>
            <div className="descriptor-item">
              <div className="descriptor-label">Width (mm)</div>
              <div className="descriptor-value">8.7</div>
              <div className="descriptor-bar">
                <div className="bar-fill" style={{width: '65%'}}></div>
              </div>
            </div>
            <div className="descriptor-item">
              <div className="descriptor-label">Aspect Ratio</div>
              <div className="descriptor-value">1.43</div>
              <div className="descriptor-bar">
                <div className="bar-fill" style={{width: '71%'}}></div>
              </div>
            </div>
            <div className="descriptor-item">
              <div className="descriptor-label">Roundness</div>
              <div className="descriptor-value">0.68</div>
              <div className="descriptor-bar">
                <div className="bar-fill" style={{width: '88%'}}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Otolith;
