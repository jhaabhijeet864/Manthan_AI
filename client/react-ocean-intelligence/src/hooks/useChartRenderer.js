import { useEffect, useRef } from 'react';

// Chart rendering functions adapted for React
const useChartRenderer = () => {
  const renderChart = (canvas, chartType) => {
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    // Set canvas size properly
    canvas.width = width;
    canvas.height = height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    try {
      switch (chartType) {
        case 'sstTrendChart':
        case 'sstChart':
          drawAnimatedLineChart(ctx, width, height, generateTemperatureData(), 'SST Trends', '°C');
          break;
        case 'salinityChart':
          drawAnimatedAreaChart(ctx, width, height, generateSalinityData(), 'Salinity Distribution', '‰');
          break;
        case 'catchChart':
          drawAnimatedPieChart(ctx, width, height, generateCatchData());
          break;
        case 'effortChart':
          drawAnimatedBarChart(ctx, width, height, generateEffortData());
          break;
        default:
          drawRealtimeWaveChart(ctx, width, height);
      }
    } catch (error) {
      console.error('Chart rendering error:', error);
      drawFallbackChart(ctx, width, height, chartType);
    }
  };

  // Data generation functions
  const generateTemperatureData = () => {
    const data = [];
    let temp = 28;
    for (let i = 0; i < 24; i++) {
      temp += (Math.random() - 0.5) * 2;
      temp = Math.max(26, Math.min(32, temp));
      data.push(temp);
    }
    return data;
  };

  const generateSalinityData = () => {
    const data = [];
    let salinity = 35.2;
    for (let i = 0; i < 30; i++) {
      salinity += (Math.random() - 0.5) * 0.4 + Math.cos(i * 0.2) * 0.3;
      salinity = Math.max(33.5, Math.min(36.8, salinity));
      data.push({ value: salinity, timestamp: i });
    }
    return data;
  };

  const generateCatchData = () => {
    return [
      { label: 'Tuna', value: 35, color: '#ff6b6b' },
      { label: 'Sardine', value: 28, color: '#4ecdc4' },
      { label: 'Mackerel', value: 18, color: '#45b7d1' },
      { label: 'Pomfret', value: 12, color: '#96ceb4' },
      { label: 'Anchovy', value: 7, color: '#ffeaa7' }
    ];
  };

  const generateEffortData = () => {
    return [
      { label: 'Jan', effort: 85, yield: 72, bycatch: 8 },
      { label: 'Feb', effort: 78, yield: 68, bycatch: 6 },
      { label: 'Mar', effort: 92, yield: 85, bycatch: 12 },
      { label: 'Apr', effort: 88, yield: 79, bycatch: 9 },
      { label: 'May', effort: 96, yield: 88, bycatch: 14 },
      { label: 'Jun', effort: 82, yield: 75, bycatch: 7 }
    ];
  };

  // Chart drawing functions
  const drawAnimatedLineChart = (ctx, width, height, data, title, unit) => {
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding - 30;
    const minValue = Math.min(...data);
    const maxValue = Math.max(...data);
    const range = maxValue - minValue;
    
    // Background
    ctx.fillStyle = 'rgba(0, 20, 40, 0.3)';
    ctx.fillRect(0, 0, width, height);
    
    // Title
    ctx.fillStyle = '#00ffff';
    ctx.font = '14px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 20);
    
    // Grid lines
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const y = padding + 30 + (i * chartHeight / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Y-axis labels
      const value = maxValue - (i * range / 5);
      ctx.fillStyle = '#8892b0';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(1) + unit, padding - 5, y + 3);
    }
    
    // Data line with glow effect
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#00ffff');
    gradient.addColorStop(0.5, '#0080ff');
    gradient.addColorStop(1, '#00ffff');
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 3;
    ctx.shadowColor = '#00ffff';
    ctx.shadowBlur = 8;
    
    ctx.beginPath();
    data.forEach((value, index) => {
      const x = padding + (index * chartWidth / (data.length - 1));
      const y = padding + 30 + ((maxValue - value) / range) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Data points
    data.forEach((value, index) => {
      const x = padding + (index * chartWidth / (data.length - 1));
      const y = padding + 30 + ((maxValue - value) / range) * chartHeight;
      
      ctx.fillStyle = '#00ffff';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Highlight every 6th point
      if (index % 6 === 0) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  const drawAnimatedAreaChart = (ctx, width, height, data, title, unit) => {
    const padding = 45;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding - 30;
    const minValue = Math.min(...data.map(d => d.value));
    const maxValue = Math.max(...data.map(d => d.value));
    const range = maxValue - minValue;
    
    // Background
    ctx.fillStyle = 'rgba(20, 30, 60, 0.2)';
    ctx.fillRect(0, 0, width, height);
    
    // Title
    ctx.fillStyle = '#4834d4';
    ctx.font = '14px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 20);
    
    // Grid
    ctx.strokeStyle = 'rgba(72, 52, 212, 0.2)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const y = padding + 30 + (i * chartHeight / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Labels
      const value = maxValue - (i * range / 5);
      ctx.fillStyle = '#8892b0';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(1) + unit, padding - 5, y + 3);
    }
    
    // Area gradient
    const areaGradient = ctx.createLinearGradient(0, padding + 30, 0, height - padding);
    areaGradient.addColorStop(0, 'rgba(72, 52, 212, 0.4)');
    areaGradient.addColorStop(0.7, 'rgba(104, 109, 224, 0.2)');
    areaGradient.addColorStop(1, 'rgba(72, 52, 212, 0.05)');
    
    ctx.fillStyle = areaGradient;
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
    
    // Top line with glow
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
    ctx.shadowBlur = 0;
    
    // Data points
    data.forEach((point, index) => {
      if (index % 3 === 0) {
        const x = padding + (index * chartWidth / (data.length - 1));
        const y = padding + 30 + ((maxValue - point.value) / range) * chartHeight;
        
        ctx.fillStyle = '#686de0';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  const drawAnimatedPieChart = (ctx, width, height, data) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 30;
    const innerRadius = radius * 0.4;
    
    // Background circle
    ctx.fillStyle = 'rgba(30, 42, 58, 0.3)';
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius + 10, 0, Math.PI * 2);
    ctx.fill();
    
    // Title
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '14px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText('Catch Composition', centerX, 25);
    
    let currentAngle = -Math.PI / 2;
    const total = data.reduce((sum, item) => sum + item.value, 0);
    
    data.forEach((item, index) => {
      const sliceAngle = (item.value / total) * 2 * Math.PI;
      
      // Main slice
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
      ctx.closePath();
      ctx.fill();
      
      // Glow effect
      ctx.shadowColor = item.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Stroke
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Labels
      const labelAngle = currentAngle + sliceAngle / 2;
      const labelRadius = radius + 25;
      const labelX = centerX + Math.cos(labelAngle) * labelRadius;
      const labelY = centerY + Math.sin(labelAngle) * labelRadius;
      
      ctx.fillStyle = '#ffffff';
      ctx.font = '11px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(item.label, labelX, labelY);
      
      // Percentage
      ctx.fillStyle = item.color;
      ctx.font = 'bold 10px JetBrains Mono';
      ctx.fillText(`${item.value}%`, labelX, labelY + 12);
      
      currentAngle += sliceAngle;
    });
    
    // Center circle with total
    ctx.fillStyle = 'rgba(10, 14, 26, 0.9)';
    ctx.beginPath();
    ctx.arc(centerX, centerY, innerRadius - 5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Center text
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 14px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('100%', centerX, centerY - 5);
    
    ctx.fillStyle = '#8892b0';
    ctx.font = '10px Lexend';
    ctx.fillText('Total Catch', centerX, centerY + 10);
  };

  const drawAnimatedBarChart = (ctx, width, height, data) => {
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding - 60;
    const barWidth = Math.max(20, chartWidth / (data.length * 4));
    
    // Background
    ctx.fillStyle = 'rgba(40, 50, 70, 0.2)';
    ctx.fillRect(0, 0, width, height);
    
    // Title
    ctx.fillStyle = '#ffa726';
    ctx.font = 'bold 16px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText('Monthly Fishing Analytics', width / 2, 30);
    
    const maxValue = Math.max(...data.map(d => Math.max(d.effort, d.yield, d.bycatch)));
    
    // Grid lines
    ctx.strokeStyle = 'rgba(255, 167, 38, 0.15)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const y = padding + 50 + (i * chartHeight / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Y-axis labels
      const value = Math.round(maxValue - (i * maxValue / 5));
      ctx.fillStyle = '#8892b0';
      ctx.font = '12px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(value.toString(), padding - 10, y + 4);
    }
    
    data.forEach((item, index) => {
      const baseX = padding + (index + 0.5) * (chartWidth / data.length);
      const barSpacing = barWidth * 0.2;
      
      // Effort bar (orange)
      const effortHeight = (item.effort / maxValue) * chartHeight;
      const effortGradient = ctx.createLinearGradient(0, height - padding - effortHeight, 0, height - padding);
      effortGradient.addColorStop(0, '#ff7043');
      effortGradient.addColorStop(1, '#ffa726');
      
      ctx.fillStyle = effortGradient;
      const effortX = baseX - barWidth - barSpacing;
      ctx.fillRect(effortX, height - padding - 50 - effortHeight, barWidth, effortHeight);
      
      ctx.shadowColor = '#ffa726';
      ctx.shadowBlur = 6;
      ctx.fillRect(effortX, height - padding - 50 - effortHeight, barWidth, effortHeight);
      ctx.shadowBlur = 0;
      
      // Yield bar (cyan)
      const yieldHeight = (item.yield / maxValue) * chartHeight;
      const yieldGradient = ctx.createLinearGradient(0, height - padding - yieldHeight, 0, height - padding);
      yieldGradient.addColorStop(0, '#0080ff');
      yieldGradient.addColorStop(1, '#00ffff');
      
      ctx.fillStyle = yieldGradient;
      const yieldX = baseX;
      ctx.fillRect(yieldX, height - padding - 50 - yieldHeight, barWidth, yieldHeight);
      
      ctx.shadowColor = '#00ffff';
      ctx.shadowBlur = 6;
      ctx.fillRect(yieldX, height - padding - 50 - yieldHeight, barWidth, yieldHeight);
      ctx.shadowBlur = 0;
      
      // Bycatch bar (red)
      const bycatchHeight = (item.bycatch / maxValue) * chartHeight;
      const bycatchGradient = ctx.createLinearGradient(0, height - padding - bycatchHeight, 0, height - padding);
      bycatchGradient.addColorStop(0, '#ee5a24');
      bycatchGradient.addColorStop(1, '#ff6b6b');
      
      ctx.fillStyle = bycatchGradient;
      const bycatchX = baseX + barWidth + barSpacing;
      ctx.fillRect(bycatchX, height - padding - 50 - bycatchHeight, barWidth * 0.7, bycatchHeight);
      
      ctx.shadowColor = '#ff6b6b';
      ctx.shadowBlur = 4;
      ctx.fillRect(bycatchX, height - padding - 50 - bycatchHeight, barWidth * 0.7, bycatchHeight);
      ctx.shadowBlur = 0;
      
      // X-axis labels
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 13px Outfit';
      ctx.textAlign = 'center';
      ctx.fillText(item.label, baseX, height - padding + 25);
    });
  };

  const drawRealtimeWaveChart = (ctx, width, height) => {
    // Simple wave chart fallback
    ctx.fillStyle = 'rgba(0, 40, 80, 0.2)';
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let x = 0; x < width; x++) {
      const y = height / 2 + Math.sin(x * 0.02 + Date.now() * 0.005) * 20;
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  const drawFallbackChart = (ctx, width, height, chartType) => {
    ctx.fillStyle = 'rgba(30, 42, 58, 0.4)';
    ctx.fillRect(0, 0, width, height);
    
    ctx.fillStyle = '#8892b0';
    ctx.font = '14px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${chartType} Chart`, width / 2, height / 2);
    ctx.fillText('Loading...', width / 2, height / 2 + 20);
  };

  return { renderChart };
};

export default useChartRenderer;
