import React from 'react';

export default function IndicatorPanel({ title, lines }) {
  return (
    <div className="panel">
      <div className="title">{title}</div>
      <pre>
        {lines.map((ln, i) => (
          <span key={i} dangerouslySetInnerHTML={{ __html: ln + '\n' }} />
        ))}
      </pre>
    </div>
  );
}
