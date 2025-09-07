import React, { useState } from 'react';

const Taxonomy = ({ isActive }) => {
  const [expandedNodes, setExpandedNodes] = useState(new Set());

  const toggleNode = (nodeId) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const TreeNode = ({ node, level = 0 }) => {
    const isExpanded = expandedNodes.has(node.id);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div className="tree-node" style={{'--level': level}}>
        <div 
          className="node-content"
          onClick={() => hasChildren && toggleNode(node.id)}
        >
          {hasChildren && (
            <div className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
              <i className="fas fa-plus"></i>
            </div>
          )}
          <div className="node-icon">
            <i className={node.icon}></i>
          </div>
          <div className="node-label">{node.label}</div>
          <div className="node-count">{node.count}</div>
        </div>
        {hasChildren && isExpanded && (
          <div className="tree-children">
            {node.children.map(child => (
              <TreeNode key={child.id} node={child} level={level + 1} />
            ))}
          </div>
        )}
      </div>
    );
  };

  const taxonomyData = {
    id: 'root',
    label: 'Marine Life',
    icon: 'fas fa-water',
    count: '2,847 species',
    children: [
      {
        id: 'chordata',
        label: 'Chordata',
        icon: 'fas fa-fish',
        count: '1,234 species',
        children: [
          {
            id: 'actinopterygii',
            label: 'Actinopterygii',
            icon: 'fas fa-fish',
            count: '856 species'
          },
          {
            id: 'chondrichthyes',
            label: 'Chondrichthyes',
            icon: 'fas fa-fish',
            count: '147 species'
          }
        ]
      },
      {
        id: 'cnidaria',
        label: 'Cnidaria',
        icon: 'fas fa-circle',
        count: '423 species',
        children: [
          {
            id: 'anthozoa',
            label: 'Anthozoa',
            icon: 'fas fa-circle',
            count: '289 species'
          }
        ]
      },
      {
        id: 'mollusca',
        label: 'Mollusca',
        icon: 'fas fa-shell',
        count: '678 species'
      },
      {
        id: 'arthropoda',
        label: 'Arthropoda',
        icon: 'fas fa-bug',
        count: '512 species'
      }
    ]
  };

  return (
    <section className={`content-section ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2><i className="fas fa-sitemap"></i> Taxonomic Classification</h2>
        <div className="section-indicator"></div>
      </div>
      
      <div className="taxonomy-container">
        <div className="taxonomy-tree">
          <TreeNode node={taxonomyData} />
        </div>
      </div>
    </section>
  );
};

export default Taxonomy;
