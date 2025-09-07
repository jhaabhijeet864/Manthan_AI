import React from 'react';

const Navigation = ({ activeSection, onSectionChange }) => {
  const navItems = [
    { id: 'overview', icon: 'fas fa-chart-line', label: 'Overview' },
    { id: 'oceanography', icon: 'fas fa-thermometer-half', label: 'Oceanography' },
    { id: 'fisheries', icon: 'fas fa-fish', label: 'Fisheries' },
    { id: 'biodiversity', icon: 'fas fa-dna', label: 'Biodiversity (eDNA)' },
    { id: 'otolith', icon: 'fas fa-microscope', label: 'Otolith Morphology' },
    { id: 'taxonomy', icon: 'fas fa-sitemap', label: 'Taxonomy' }
  ];

  return (
    <nav className="main-nav">
      <div className="nav-container">
        {navItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => onSectionChange(item.id)}
          >
            <i className={item.icon}></i>
            <span>{item.label}</span>
            <div className="nav-glow"></div>
          </div>
        ))}
      </div>
    </nav>
  );
};

export default Navigation;
