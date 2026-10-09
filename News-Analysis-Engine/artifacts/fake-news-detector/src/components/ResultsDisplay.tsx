import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, ShieldAlert, Clock, Link as LinkIcon, Activity } from 'lucide-react';
import { cn, formatConfidence, getScoreColorClass, getScoreBgColorClass, getScoreGlowClass } from '@/lib/utils';
import { ScoreGauge } from './ScoreGauge';
import { MarkdownText } from './MarkdownText';

// Use any to bypass strict type checking if barrel export isn't perfectly mapped, 
// but define interface locally for safety based on OpenAPI spec provided.
interface ModelScore {
  modelName: string;
  modelType: string;
  score: number;
  label: string;
  details: string;
  conceptScores?: number[];
  cbllm_concept_probs?: number[];
  cbllm_concept_names?: string[];
  topConcepts?: Array<{
    rank: number;
    name: string;
    score: number;
    index: number;
    verified?: boolean;
  }>;
}

interface DetectionResult {
  id: string;
  title: string;
  sourceUrl?: string;
  textAigcScore: ModelScore;
  textIntentScore: ModelScore;
  imageAigcScore: ModelScore;
  finalScore: number;
  finalLabel: string;
  report: string;
  analyzedAt: string;
}

interface ResultsDisplayProps {
  result: DetectionResult | null;
  isLoading: boolean;
}

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 300, damping: 24 } }
};

export function ResultsDisplay({ result, isLoading }: ResultsDisplayProps) {
  // Declare all hooks at the top level (React Hooks rules)
  const [openTextIntent, setOpenTextIntent] = useState(false);
  const [openImageConcepts, setOpenImageConcepts] = useState(false);
  
  if (isLoading) {
    return (
      <div className="w-full h-96 flex flex-col items-center justify-center glass-panel rounded-3xl p-8 border border-white/5">
        <motion.div
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="relative"
        >
          <div className="w-24 h-24 rounded-full border-4 border-primary/30 border-t-primary animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Activity className="w-8 h-8 text-primary" />
          </div>
        </motion.div>
        <h3 className="mt-8 text-xl font-display text-white animate-pulse text-glow">
          Running Multi-Model Analysis...
        </h3>
        <p className="mt-2 text-muted-foreground text-center max-w-md">
          Passing content through RoBERTa (AIGC), CB-LLM (Intent), and CBM (Image) fusion pipelines.
        </p>
      </div>
    );
  }

  if (!result) return null;

  const isFake = result.finalScore > 0.66;
  const isReal = result.finalScore < 0.33;
  
  const StatusIcon = isFake ? AlertTriangle : (isReal ? CheckCircle2 : ShieldAlert);
  const statusColor = getScoreColorClass(result.finalScore);
  const statusBg = getScoreBgColorClass(result.finalScore);
  const statusGlow = getScoreGlowClass(result.finalScore);

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="w-full flex flex-col gap-6"
    >
      {/* FINAL VERDICT BANNER */}
      <motion.div 
        variants={itemVariants}
        className={cn(
          "w-full rounded-3xl p-8 flex flex-col md:flex-row items-center justify-between gap-6 border border-white/10 relative overflow-hidden",
          "glass-panel",
          statusGlow
        )}
      >
        <div className={cn("absolute inset-0 opacity-10 pointer-events-none", statusBg)} />
        
        <div className="flex items-center gap-6 z-10">
          <div className={cn("p-4 rounded-2xl bg-black/40 border border-white/5 backdrop-blur-md", statusColor)}>
            <StatusIcon className="w-12 h-12" />
          </div>
          <div>
            <p className="text-sm font-medium text-white/60 tracking-wider uppercase mb-1">Final Fusion Verdict</p>
            <h2 className={cn("text-4xl md:text-5xl font-display font-bold uppercase text-glow", statusColor)}>
              {result.finalLabel}
            </h2>
          </div>
        </div>
        
        <div className="flex flex-col items-end z-10 text-right gap-4">
          {/* Primary: Fake Probability Score */}
          <div>
            <div className="text-sm text-white/60 mb-1">Fake Probability</div>
            <div className="text-4xl font-mono font-bold text-white">
              {formatConfidence(result.finalScore)}
            </div>
          </div>
        </div>
      </motion.div>

      {/* INDIVIDUAL MODEL SCORES */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { key: 'textAigcScore', title: 'Text AIGC', desc: 'RoBERTa-based detection' },
          { key: 'textIntentScore', title: 'Text Intent', desc: 'CB-LLM semantic analysis' },
          { key: 'imageAigcScore', title: 'Image AIGC', desc: 'CBM visual artifacts' }
        ].map((model, idx) => {
          const scoreData = result[model.key as keyof DetectionResult] as ModelScore;
          if (!scoreData) return null;
          
          const isNA = scoreData.score === -1 || scoreData.label === 'N/A';
          const displayScore = isNA ? 0 : scoreData.score;

          return (
            <motion.div 
              key={model.key}
              variants={itemVariants}
              className="glass-panel rounded-2xl p-6 border border-white/5 flex flex-col relative overflow-hidden group hover:border-white/10 transition-colors"
            >
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="font-display font-semibold text-lg text-white">{model.title}</h3>
                  <p className="text-xs text-muted-foreground">{model.desc}</p>
                  <div className="mt-2 space-y-0.5">
                    <p className="text-xs text-white/90 font-semibold">
                      {isNA ? 'Score: N/A' : `Score: ${displayScore.toFixed(4)}`}
                    </p>
                  </div>
                </div>
                <div className={cn("px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider bg-black/50 border border-white/5", getScoreColorClass(displayScore))}>
                  {isNA ? 'N/A' : scoreData.label}
                </div>
              </div>
              
              <div className="flex-1 flex flex-col items-center justify-center my-4">
                <ScoreGauge 
                  score={displayScore} 
                  label="Probability" 
                  className={isNA ? 'opacity-30 grayscale' : ''} 
                />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* TEXT INTENT CONCEPT EXPLANATION (Collapsible) */}
      {result.textIntentScore && result.textIntentScore.cbllm_concept_probs && result.textIntentScore.cbllm_concept_probs.length > 0 && ((() => {
          const intentScores = result.textIntentScore.cbllm_concept_probs ?? [];
          const intentNames = result.textIntentScore.cbllm_concept_names ?? [];
          const formatIntentName = (name: string) => name.replace(/^new_/, '');
          const wrapEveryTwoChars = (name: string) => {
            const cleaned = formatIntentName(name);
            const chunks = cleaned.match(/.{1,2}/g);
            return (chunks ?? [cleaned]).join('\n');
          };
          const rankedConcepts = intentScores
            .map((score, index) => ({
              index,
              score,
              name: formatIntentName(intentNames[index] ?? `concept_${index + 1}`),
              wrappedName: wrapEveryTwoChars(intentNames[index] ?? `concept_${index + 1}`),
            }))
            .sort((a, b) => b.score - a.score);
          const maxIntentScore = Math.max(...rankedConcepts.map((c) => c.score), 1e-6);

          return (
            <motion.div variants={itemVariants} className="glass-panel rounded-3xl p-6 md:p-8 border border-white/5">
              <button
                className="flex items-center gap-2 w-full text-left font-display font-semibold text-xl text-white mb-6 focus:outline-none"
                onClick={() => setOpenTextIntent((v) => !v)}
                aria-expanded={openTextIntent}
              >
                <Activity className="w-5 h-5 text-primary" />
                Text Intent Concept Breakdown
                <span className="ml-auto text-base text-primary">{openTextIntent ? '▲' : '▼'}</span>
              </button>

              {openTextIntent && (
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-white/80 mb-3">15-Dimensional Intent Concept Distribution (High to Low)</h4>
                    <div className="p-4 rounded-lg bg-black/40 border border-white/5">
                      <div className="text-xs text-white/60 mb-2">
                        Sorted by concept score. Lower overall values indicate weaker intent signals.
                      </div>
                      <div className="overflow-x-auto pb-2">
                        <div
                          className="grid gap-1 min-w-[720px]"
                          style={{ gridTemplateColumns: `repeat(${rankedConcepts.length}, minmax(44px, 1fr))` }}
                        >
                          {rankedConcepts.map((concept, i) => (
                            <div key={`${concept.name}-${concept.index}`} className="flex flex-col items-center">
                              {(() => {
                                const normalized = concept.score / maxIntentScore;
                                const visualHeight = 10 + Math.pow(normalized, 0.5) * 90;
                                return (
                                  <>
                              <div className="text-[10px] text-white/70 mb-1">{(concept.score * 100).toFixed(2)}%</div>
                              <div className="w-full h-28 flex items-end">
                                <div
                                  className="w-full bg-gradient-to-t from-primary/50 to-primary rounded-t-sm hover:from-primary hover:to-accent transition-all"
                                  style={{ height: `${Math.min(100, visualHeight)}%` }}
                                  title={`#${i + 1} ${concept.name}: ${(concept.score * 100).toFixed(2)}%`}
                                />
                              </div>
                              <div className="text-[10px] text-primary mt-1">#{i + 1}</div>
                              <div className="text-[10px] text-white/75 leading-tight text-center whitespace-pre-line break-keep min-h-[2.8rem] mt-1">
                                {concept.wrappedName}
                              </div>
                                  </>
                                );
                              })()}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="mt-3 text-right text-xs text-white/60">
                        <div>Min: {(Math.min(...intentScores) * 100).toFixed(2)}%</div>
                        <div>Avg: {((intentScores.reduce((a: number, b: number) => a + b, 0) / Math.max(1, intentScores.length)) * 100).toFixed(2)}%</div>
                        <div>Max: {(Math.max(...intentScores) * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          );
        })())
      }

      {/* IMAGE AIGC CONCEPT EXPLANATION */}
      {/* IMAGE AIGC CONCEPT EXPLANATION (Collapsible) */}
      {/* Show when topConcepts are available (both low and high score scenarios) */}
      {result.imageAigcScore && result.imageAigcScore.topConcepts && result.imageAigcScore.topConcepts.length > 0 && ((() => {
          // 根據分數判斷是低分還是高分情境
          const isLowScore = result.imageAigcScore.score < 0.45;
          const scoreDesc = isLowScore ? 'Low-Score Features' : 'High-Score Features';
          const featureDesc = isLowScore 
            ? 'These are the lowest-scoring concepts detected by CBM, representing features more typical of real images.'
            : 'These are the highest-scoring concepts detected by CBM, representing features more typical of AI-generated images.';
          // 檢查是否有 verified 字段（不論 true 或 false）
          const hasOwlv2Verification = result.imageAigcScore.topConcepts[0]?.verified !== undefined;
          // 只顯示驗證通過（verified: true）的概念
          const topConcepts = result.imageAigcScore.topConcepts.filter(c => c.verified === true).slice(0, 10);
          return (
            <motion.div variants={itemVariants} className="glass-panel rounded-3xl p-6 md:p-8 border border-white/5">
              <button
                className="flex items-center gap-2 w-full text-left font-display font-semibold text-xl text-white mb-6 focus:outline-none"
                onClick={() => setOpenImageConcepts((v) => !v)}
                aria-expanded={openImageConcepts}
              >
                <Activity className="w-5 h-5 text-primary" />
                Image AIGC Concept Breakdown
                <span className="ml-auto text-base text-primary">{openImageConcepts ? '▲' : '▼'}</span>
              </button>
              {openImageConcepts && (
                <div className="space-y-4">
                  {hasOwlv2Verification && (
                    <div>
                      <h4 className="text-sm font-semibold text-white/80 mb-3">
                        {scoreDesc}
                      </h4>
                      <div className="text-xs text-white/60 mb-3">
                        {featureDesc}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {topConcepts.map((concept) => (
                          <div 
                            key={concept.index}
                            className="p-3 rounded-lg bg-black/40 border border-white/5 hover:border-white/10 transition-colors"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <span className="text-sm font-medium text-white/80 flex-1">{concept.name}</span>
                              <span className="text-xs font-mono font-bold text-primary ml-2">#{concept.rank}</span>
                            </div>
                            <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all"
                                style={{ width: `${Math.min(concept.score * 100, 100)}%` }}
                              />
                            </div>
                            <div className="text-xs text-white/60 mt-1">{(concept.score * 100).toFixed(2)}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          );
        })())
      }

      {/* LLM REPORT */}
      <motion.div variants={itemVariants} className="glass-panel rounded-3xl p-6 md:p-8 border border-white/5">
        <h3 className="font-display font-semibold text-xl text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          Deep Analysis Report
        </h3>
        <div className="prose prose-invert max-w-none">
          <MarkdownText 
            content={result.report}
            className="p-6 rounded-2xl bg-black/40 border border-white/5 text-white/90 leading-relaxed font-sans text-sm md:text-base"
          />
        </div>
      </motion.div>

      {/* METADATA */}
      <motion.div variants={itemVariants} className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground pt-2">
        <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-full border border-white/5">
          <Clock className="w-3.5 h-3.5" />
          {new Date(result.analyzedAt).toLocaleString()}
        </div>
        {result.sourceUrl && (
          <a 
            href={result.sourceUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-full border border-white/5 hover:bg-white/10 transition-colors text-primary"
          >
            <LinkIcon className="w-3.5 h-3.5" />
            Source URL
          </a>
        )}
        <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-full border border-white/5">
          ID: <span className="font-mono">{result.id}</span>
        </div>
      </motion.div>

    </motion.div>
  );
}
