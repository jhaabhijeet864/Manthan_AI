import React from 'react';

const Biodiversity = ({ isActive }) => {
  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2><i className="fas fa-leaf"></i> Biodiversity Analysis</h2>
        <div className="section-indicator"></div>
      </div>
      
      <div className="search-panel">
        <div className="search-container">
          <div className="search-input-wrapper">
            <i className="fas fa-search"></i>
            <input 
              type="text" 
              className="species-search" 
              placeholder="Search species by eDNA sequence, common name, or location..."
            />
            <button className="search-btn">
              <i className="fas fa-microscope"></i>
            </button>
          </div>
          <div className="search-filters">
            <button className="filter-btn active">All Species</button>
            <button className="filter-btn">Fish</button>
            <button className="filter-btn">Coral</button>
            <button className="filter-btn">Plankton</button>
            <button className="filter-btn">Endangered</button>
          </div>
        </div>
      </div>

      <div className="biodiversity-grid">
        <div className="matches-panel">
          <div className="panel-header">
            <h3><i className="fas fa-dna"></i> eDNA Analysis Results</h3>
            <button className="upload-btn">
              <i className="fas fa-upload"></i>
              Upload Sample
            </button>
          </div>
          <div className="matches-list">
            <div className="match-item">
              <div className="match-species">Epinephelus marginatus</div>
              <div className="match-common">Dusky Grouper</div>
              <div className="match-confidence">
                <div className="confidence-bar" style={{'--width': '94%'}}></div>
                <span>94.2%</span>
              </div>
              <div className="match-location">Arabian Sea, 150m depth</div>
            </div>
            
            <div className="match-item">
              <div className="match-species">Thalassoma hardwicke</div>
              <div className="match-common">Sixbar Wrasse</div>
              <div className="match-confidence">
                <div className="confidence-bar" style={{'--width': '87%'}}></div>
                <span>87.6%</span>
              </div>
              <div className="match-location">Lakshadweep Sea, 45m depth</div>
            </div>
          </div>
        </div>
        
        <div className="stats-panel">
          <div className="panel-header">
            <h3><i className="fas fa-chart-bar"></i> Diversity Statistics</h3>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">2,847</div>
              <div className="stat-label">Total Species</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">156</div>
              <div className="stat-label">New Discoveries</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">23</div>
              <div className="stat-label">Endangered</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">94.2%</div>
              <div className="stat-label">ID Accuracy</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Biodiversity;
