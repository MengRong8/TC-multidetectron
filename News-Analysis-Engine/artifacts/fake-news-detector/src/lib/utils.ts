import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

export function getScoreColorClass(score: number): string {
  // Assuming 0 is Real (green), 1 is Fake (red)
  if (score < 0.33) return "text-success";
  if (score < 0.66) return "text-warning";
  return "text-destructive";
}

export function getScoreBgColorClass(score: number): string {
  if (score < 0.33) return "bg-success";
  if (score < 0.66) return "bg-warning";
  return "bg-destructive";
}

export function getScoreGlowClass(score: number): string {
  if (score < 0.33) return "glow-success";
  if (score < 0.66) return "glow-warning";
  return "glow-destructive";
}
