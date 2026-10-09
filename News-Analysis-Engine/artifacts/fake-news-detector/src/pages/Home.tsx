import React, { useState, useEffect } from 'react';
import { InputForm } from '@/components/InputForm';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { useToast } from '@/hooks/use-toast';
import { ShieldCheck, Database, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Home() {
  const [result, setResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  // Reset scroll when result comes in
  useEffect(() => {
    if (result) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
  }, [result]);

  const handleError = (msg: string) => {
    toast({
      title: "Error",
      description: msg,
      variant: "destructive",
    });
  };

  const handleResults = (data: any) => {
    setResult(data);
    setIsAnalyzing(false);
    toast({
      title: "Analysis Complete",
      description: "Successfully processed through all models.",
      style: { backgroundColor: 'hsl(var(--success))', color: 'white', border: 'none' }
    });
  };

  return (
    <div className="relative min-h-screen pb-24">
      {/* Background Image */}
      <div 
        className="fixed inset-0 z-[-1] bg-cover bg-center bg-no-repeat opacity-40 mix-blend-screen"
        style={{ backgroundImage: `url(${import.meta.env.BASE_URL}images/cyber-bg.png)` }}
      />
      
      {/* Top Gradient Overlay for readability */}
      <div className="fixed inset-0 z-[-1] bg-gradient-to-b from-background via-background/90 to-background" />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 flex flex-col gap-12">
        
        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="text-center space-y-6"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium tracking-wide">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            SYSTEM ONLINE
          </div>
          
          <h1 className="text-5xl md:text-7xl font-display font-bold text-white tracking-tight">
            TC - <span className="text-primary">MultiDetectorn</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Advanced multi-modal fake news detection. Powered by RoBERTa, CB-LLM, and CBM to analyze text patterns, intent, and image authenticity.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-6 pt-4 text-sm font-medium text-white/50">
            <div className="flex items-center gap-2"><Database className="w-4 h-4" /> Text AIGC</div>
            <div className="flex items-center gap-2"><Zap className="w-4 h-4" /> Intent Analysis</div>
            <div className="flex items-center gap-2"><ShieldCheck className="w-4 h-4" /> Visual Verification</div>
          </div>
        </motion.div>

        {/* Form Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
        >
          <InputForm 
            onResults={handleResults} 
            onLoadingChange={setIsAnalyzing}
            onError={handleError}
          />
        </motion.div>

        {/* Results Section */}
        {(isAnalyzing || result) && (
          <div className="pt-8 border-t border-white/10">
             <ResultsDisplay result={result} isLoading={isAnalyzing} />
          </div>
        )}

      </main>
    </div>
  );
}
