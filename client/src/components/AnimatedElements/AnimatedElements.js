import React from 'react';

export const FloatingIconMini = ({ icon }) => (
  <div className="floating-icon-mini">
    <i className={icon}></i>
  </div>
);

export const FloatingMarineIcon = ({ icon, delay = 0 }) => (
  <div className={`floating-marine-icon ${delay ? `delay-${delay}` : ''}`}>
    <i className={icon}></i>
  </div>
);

export const FloatingDataIcon = ({ icon, delay = 0 }) => (
  <div className={`floating-data-icon ${delay ? `delay-${delay}` : ''}`}>
    <i className={icon}></i>
  </div>
);

export const DataStreamLine = ({ style, delay = 0 }) => (
  <div 
    className={`data-stream-line ${delay ? `delay-${delay}` : ''}`}
    style={style}
  />
);

export const PulseDot = ({ delay = 0 }) => (
  <div className={`pulse-dot ${delay ? `delay-${delay}` : ''}`}></div>
);

export const BlinkingDot = ({ delay = 0 }) => (
  <div className={`blinking-dot ${delay ? `delay-${delay}` : ''}`}></div>
);

export const SchoolOfFish = () => (
  <div className="floating-school-fish">
    <i className="fas fa-fish"></i>
    <i className="fas fa-fish"></i>
    <i className="fas fa-fish"></i>
    <i className="fas fa-fish"></i>
    <i className="fas fa-fish"></i>
  </div>
);
