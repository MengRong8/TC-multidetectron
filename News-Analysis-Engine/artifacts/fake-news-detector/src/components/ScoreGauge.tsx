import React from 'react';
import { motion } from 'framer-motion';
import { cn, getScoreColorClass } from '@/lib/utils';

interface ScoreGaugeProps {
  score: number; // 0 to 1
  label: string;
  className?: string;
}

export function ScoreGauge({ score, label, className }: ScoreGaugeProps) {
  const percentage = Math.round(score * 100);
  const colorClass = getScoreColorClass(score);
  
  // Convert text-x to stroke-x manually for simplicity, or just use CSS variables if integrated.
  // We'll use inline styles for the stroke color based on score thresholds for SVG.
  let strokeColor = 'hsl(var(--destructive))';
  if (score < 0.33) strokeColor = 'hsl(var(--success))';
  else if (score < 0.66) strokeColor = 'hsl(var(--warning))';

  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className={cn("flex flex-col items-center justify-center relative", className)}>
      <div className="relative w-24 h-24 flex items-center justify-center">
        {/* Background Circle */}
        <svg className="absolute w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-white/5"
          />
          {/* Progress Circle */}
          <motion.circle
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.2 }}
            cx="50"
            cy="50"
            r={radius}
            stroke={strokeColor}
            strokeWidth="8"
            fill="transparent"
            strokeLinecap="round"
            style={{
              strokeDasharray: circumference,
            }}
          />
        </svg>
        <div className="absolute flex flex-col items-center justify-center">
          <span className={cn("text-xl font-display font-bold", colorClass)}>
            {percentage}%
          </span>
        </div>
      </div>
      <span className="mt-3 text-sm font-medium text-muted-foreground tracking-wide uppercase">
        {label}
      </span>
    </div>
  );
}
