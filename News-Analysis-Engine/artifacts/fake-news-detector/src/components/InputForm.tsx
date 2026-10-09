import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileImage, Link as LinkIcon, PenTool, ArrowRight, Scan, ImagePlus, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useDetectFakeNews, useCrawlUrl } from '@workspace/api-client-react';

interface InputFormProps {
  onResults: (data: any) => void; // Using any for simplicity here to pass full DetectionResult
  onLoadingChange: (isLoading: boolean) => void;
  onError: (error: string) => void;
}

type TabType = 'manual' | 'crawl';

export function InputForm({ onResults, onLoadingChange, onError }: InputFormProps) {
  const [activeTab, setActiveTab] = useState<TabType>('manual');
  
  // Form State
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [url, setUrl] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | undefined>();
  const [extractedImages, setExtractedImages] = useState<string[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mutations
  const { mutateAsync: detectNews } = useDetectFakeNews();
  const { mutateAsync: crawlNews, isPending: isCrawling } = useCrawlUrl();

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      setImagePreview(base64String);
      setImageBase64(base64String);
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setImageBase64(undefined);
    setExtractedImages([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleCrawl = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    
    try {
      const result = await crawlNews({ data: { url } });
      // Reset previous image state to avoid carrying over stale images
      setImageFile(null);
      setImagePreview(null);
      setImageBase64(undefined);
      if (fileInputRef.current) fileInputRef.current.value = '';

      // Pre-fill manual tab
      setTitle(result.title || '');
      setContent(result.content || '');

      const hasContent = Boolean(result.content && result.content.trim());
      const hasImages = Boolean(result.imageUrls && result.imageUrls.length > 0);

      // If crawl misses content or images, do not reuse any previous image
      if (hasContent && hasImages) {
        setExtractedImages(result.imageUrls);
        // We'll use the first image URL if present, but since API needs base64 ideally, 
        // we might just pass the imageUrl in the detect request. The schema supports imageUrl!
      } else {
        setExtractedImages([]);
      }
      setActiveTab('manual');
    } catch (err: any) {
      onError(err.message || 'Failed to crawl URL');
    }
  };

  const handleAnalyze = async () => {
    if (!title.trim() || !content.trim()) {
      onError('Title and content are required for analysis.');
      return;
    }

    onLoadingChange(true);
    try {
      const reqData: any = {
        title,
        content,
      };
      
      if (imageBase64) {
        reqData.imageBase64 = imageBase64;
      } else if (extractedImages.length > 0) {
        reqData.imageUrl = extractedImages[0];
      }
      
      if (url) {
        reqData.sourceUrl = url;
      }

      const result = await detectNews({ data: reqData });
      onResults(result);
    } catch (err: any) {
      onError(err.message || 'Failed to analyze content');
      onLoadingChange(false);
    }
  };

  return (
    <div className="w-full glass-panel rounded-3xl overflow-hidden border border-white/10 shadow-2xl">
      {/* Tabs */}
      <div className="flex border-b border-white/10">
        <button
          onClick={() => setActiveTab('manual')}
          className={cn(
            "flex-1 py-4 flex items-center justify-center gap-2 text-sm font-medium transition-all",
            activeTab === 'manual' 
              ? "bg-white/5 text-white border-b-2 border-primary" 
              : "text-muted-foreground hover:bg-white/5 hover:text-white/80"
          )}
        >
          <PenTool className="w-4 h-4" />
          Manual Input
        </button>
        <button
          onClick={() => setActiveTab('crawl')}
          className={cn(
            "flex-1 py-4 flex items-center justify-center gap-2 text-sm font-medium transition-all",
            activeTab === 'crawl' 
              ? "bg-white/5 text-white border-b-2 border-primary" 
              : "text-muted-foreground hover:bg-white/5 hover:text-white/80"
          )}
        >
          <LinkIcon className="w-4 h-4" />
          URL Crawl
        </button>
      </div>

      <div className="p-6 md:p-8">
        <AnimatePresence mode="wait">
          {activeTab === 'crawl' ? (
            <motion.form 
              key="crawl"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              onSubmit={handleCrawl}
              className="flex flex-col gap-6"
            >
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/80">News Article URL</label>
                <div className="flex gap-3">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com/news/article"
                    className="flex-1 px-4 py-3 rounded-xl glass-input text-white placeholder:text-white/30 focus:outline-none"
                    required
                  />
                  <button
                    type="submit"
                    disabled={isCrawling || !url}
                    className="px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 hover:glow-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                  >
                    {isCrawling ? (
                      <div className="w-5 h-5 rounded-full border-2 border-black/20 border-t-black animate-spin" />
                    ) : (
                      <>
                        <Scan className="w-5 h-5" />
                        Extract
                      </>
                    )}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  We'll automatically extract the title, content, and hero image for you to review before analysis.
                </p>
              </div>
            </motion.form>
          ) : (
            <motion.div
              key="manual"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col gap-6"
            >
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/80">Article Title</label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter the headline..."
                  className="w-full px-4 py-3 rounded-xl glass-input text-white placeholder:text-white/30 focus:outline-none text-lg font-medium"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-white/80">Article Content</label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Paste the full article text here..."
                  className="w-full px-4 py-3 rounded-xl glass-input text-white placeholder:text-white/30 focus:outline-none min-h-[200px] resize-y font-sans leading-relaxed"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-white/80">Accompanying Image (Optional)</label>
                
                {!imagePreview && extractedImages.length === 0 ? (
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full border-2 border-dashed border-white/10 rounded-2xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer hover:border-primary/50 hover:bg-primary/5 transition-all group"
                  >
                    <div className="p-4 rounded-full bg-white/5 group-hover:bg-primary/20 transition-colors">
                      <ImagePlus className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-white/80">Click to upload an image</p>
                      <p className="text-xs text-muted-foreground mt-1">JPEG, PNG up to 10MB</p>
                    </div>
                  </div>
                ) : (
                  <div className="relative inline-block rounded-2xl overflow-hidden border border-white/10 group">
                    <img 
                      src={imagePreview || extractedImages[0]} 
                      alt="Preview" 
                      className="max-h-48 object-cover opacity-90 group-hover:opacity-100 transition-opacity" 
                    />
                    <button 
                      onClick={handleRemoveImage}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-destructive hover:text-white transition-colors backdrop-blur-md"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    {extractedImages.length > 0 && !imagePreview && (
                      <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md bg-black/60 text-xs text-white backdrop-blur-md border border-white/10">
                        Extracted via URL
                      </div>
                    )}
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </div>

              <div className="pt-4 border-t border-white/10 flex justify-end">
                <button
                  onClick={handleAnalyze}
                  disabled={!title.trim() || !content.trim()}
                  className="px-8 py-4 rounded-xl bg-white text-black font-bold text-lg hover:bg-gray-200 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none transition-all flex items-center gap-3 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                >
                  Initiate Deep Analysis
                  <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
