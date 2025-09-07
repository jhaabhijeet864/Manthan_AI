import React, { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import Navigation from './components/Navigation/Navigation';
import Overview from './components/Sections/Overview';
import Oceanography from './components/Sections/Oceanography';
import Fisheries from './components/Sections/Fisheries';
import Biodiversity from './components/Sections/Biodiversity';
import Otolith from './components/Sections/Otolith';
import Taxonomy from './components/Sections/Taxonomy';
import './App.css';

function App() {
  const [activeSection, setActiveSection] = useState('overview');

  // Initialize bubble animation on mount
  useEffect(() => {
    const bubbleContainer = document.querySelector('.bubble-container');
    if (bubbleContainer) {
      createBubbles(bubbleContainer);
    }
  }, []);

  const createBubbles = (container) => {
    const bubbleCount = 15;
    
    function createBubble() {
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      
      const size = Math.random() * 40 + 20;
      const left = Math.random() * 100;
      const duration = Math.random() * 10 + 10;
      const delay = Math.random() * 5;
      
      bubble.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${left}%;
        animation-duration: ${duration}s;
        animation-delay: ${delay}s;
      `;
      
      container.appendChild(bubble);
      
      setTimeout(() => {
        if (bubble.parentNode) {
          bubble.parentNode.removeChild(bubble);
        }
        createBubble();
      }, (duration + delay) * 1000);
    }
    
    for (let i = 0; i < bubbleCount; i++) {
      setTimeout(() => createBubble(), i * 1000);
    }
  };

  const handleSectionChange = (section) => {
    setActiveSection(section);
  };

  const renderActiveSection = () => {
    return (
      <>
        <Overview isActive={activeSection === 'overview'} />
        <Oceanography isActive={activeSection === 'oceanography'} />
        <Fisheries isActive={activeSection === 'fisheries'} />
        <Biodiversity isActive={activeSection === 'biodiversity'} />
        <Otolith isActive={activeSection === 'otolith'} />
        <Taxonomy isActive={activeSection === 'taxonomy'} />
      </>
    );
  };

  return (
    <div className="App">
      {/* Background Elements */}
      <div className="ocean-background">
        <div className="bubble-container"></div>
        <div className="wave-animation"></div>
        <div className="grid-overlay"></div>
      </div>

      <Header />
      <Navigation 
        activeSection={activeSection} 
        onSectionChange={handleSectionChange} 
      />

      <main className="main-content">
        {renderActiveSection()}
      </main>
    </div>
  );
}

export default App;
