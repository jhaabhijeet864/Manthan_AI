import React, { useState, useEffect } from 'react';

const Header = () => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-IN', {
      hour12: false,
      timeZone: 'Asia/Kolkata'
    });
  };

  return (
    <header className="main-header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo-icon">
            <i className="fas fa-water"></i>
          </div>
          <div className="branding">
            <h1>Manthan AI -Unified Ocean Intelligence</h1>
            <p>Centre for Marine Living Resources and Ecology · Ministry of Earth Sciences</p>
          </div>
        </div>
        <div className="header-stats">
          <div className="stat-item">
            <div className="stat-value" id="live-time">{formatTime(currentTime)}</div>
            <div className="stat-label">IST</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">LIVE</div>
            <div className="stat-label pulse-green"></div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
