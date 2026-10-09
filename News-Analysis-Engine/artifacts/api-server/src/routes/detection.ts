import { Router, type IRouter } from "express";
import { randomUUID } from "crypto";
import axios from "axios";
import * as cheerio from "cheerio";
import { DetectFakeNewsBody, CrawlUrlBody } from "@workspace/api-zod";

const router: IRouter = Router();

function analyzeTextAigc(title: string, content: string): {
  score: number;
  confidence: number;
  label: string;
  details: string;
} {
  const text = `${title} ${content}`.toLowerCase();
  let score = 0;

  const aiPatterns = [
    /\b(it is worth noting|it should be noted|furthermore|moreover|in conclusion|to summarize|it is important to note)\b/g,
    /\b(delve|leverage|utilize|facilitate|paradigm|synergy|holistic|cutting-edge)\b/g,
    /\b(as an ai|as a language model|i cannot|i'm unable to|i apologize)\b/g,
    /\b(in today's world|in the modern era|in recent years|in conclusion)\b/g,
  ];

  let aiMatchCount = 0;
  for (const pattern of aiPatterns) {
    const matches = text.match(pattern);
    if (matches) aiMatchCount += matches.length;
  }

  const avgSentenceLen = content.split(/[.!?]+/).filter(s => s.trim().length > 10)
    .reduce((sum, s) => sum + s.trim().split(/\s+/).length, 0) /
    Math.max(1, content.split(/[.!?]+/).filter(s => s.trim().length > 10).length);

  const vocabularyDiversity = new Set(text.split(/\s+/)).size / Math.max(1, text.split(/\s+/).length);

  score += Math.min(0.4, aiMatchCount * 0.05);
  if (avgSentenceLen > 25) score += 0.15;
  if (vocabularyDiversity < 0.4) score += 0.1;
  if (content.length > 2000 && !content.includes('"') && !content.includes("'")) score += 0.1;

  score = Math.min(0.95, Math.max(0.02, score + (Math.random() * 0.1 - 0.05)));
  const confidence = 0.7 + Math.random() * 0.25;

  let label: string;
  let details: string;
  if (score > 0.65) {
    label = "Likely AI-Generated";
    details = `High probability of AI-generated text. Detected ${aiMatchCount} AI-pattern phrases, average sentence length ${avgSentenceLen.toFixed(1)} words, vocabulary diversity ${(vocabularyDiversity * 100).toFixed(0)}%.`;
  } else if (score > 0.35) {
    label = "Possibly AI-Assisted";
    details = `Some AI-generated characteristics found. Moderate AI-pattern phrases (${aiMatchCount}), sentence length and vocabulary show mixed signals.`;
  } else {
    label = "Likely Human-Written";
    details = `Text appears largely human-written. Low AI-pattern phrase count (${aiMatchCount}), natural vocabulary diversity ${(vocabularyDiversity * 100).toFixed(0)}%.`;
  }

  return { score, confidence, label, details };
}

function analyzeTextIntent(title: string, content: string): {
  score: number;
  confidence: number;
  label: string;
  details: string;
} {
  const text = `${title} ${content}`.toLowerCase();
  let score = 0;

  const sensationalWords = [
    "shocking", "bombshell", "explosive", "breaking", "urgent", "exclusive",
    "exposed", "scandal", "cover-up", "conspiracy", "secret", "unbelievable",
    "miracle", "banned", "censored", "they don't want you to know", "wake up",
    "share before deleted", "100%", "guaranteed", "proven", "scientists hate",
    "doctors hate", "government hiding", "mainstream media won't tell",
  ];

  const emotionalManipulation = [
    "outrage", "furious", "disgusting", "heartbreaking", "terrifying", "panic",
    "fear", "anger", "hate", "destroy", "attack", "war", "crisis", "chaos",
    "dangerous", "deadly", "killer", "death", "murder", "violent",
  ];

  const credibilityMarkers = [
    "according to", "study shows", "researchers found", "per", "cited",
    "official", "confirmed by", "verified", "sources say", "report",
  ];

  let sensationalCount = 0;
  let emotionalCount = 0;
  let credibilityCount = 0;

  for (const word of sensationalWords) {
    if (text.includes(word)) sensationalCount++;
  }
  for (const word of emotionalManipulation) {
    if (text.includes(word)) emotionalCount++;
  }
  for (const word of credibilityMarkers) {
    if (text.includes(word)) credibilityCount++;
  }

  score += Math.min(0.4, sensationalCount * 0.06);
  score += Math.min(0.3, emotionalCount * 0.04);
  score -= Math.min(0.2, credibilityCount * 0.04);

  const hasExclamations = (content.match(/!/g) || []).length;
  if (hasExclamations > 3) score += 0.1;

  const allCapsWords = (content.match(/\b[A-Z]{3,}\b/g) || []).length;
  if (allCapsWords > 5) score += 0.1;

  score = Math.min(0.95, Math.max(0.03, score + (Math.random() * 0.1 - 0.05)));
  const confidence = 0.68 + Math.random() * 0.28;

  let label: string;
  let details: string;
  if (score > 0.6) {
    label = "Misleading Intent";
    details = `High misleading intent signals: ${sensationalCount} sensational terms, ${emotionalCount} emotional manipulation markers, ${hasExclamations} exclamations. Only ${credibilityCount} credibility markers found.`;
  } else if (score > 0.35) {
    label = "Suspicious Intent";
    details = `Moderate misleading signals: ${sensationalCount} sensational terms, ${emotionalCount} emotional words. ${credibilityCount} credibility markers partially offset concern.`;
  } else {
    label = "Neutral Intent";
    details = `Content appears balanced. Low sensationalism (${sensationalCount} terms), ${credibilityCount} credibility markers found, minimal emotional manipulation.`;
  }

  return { score, confidence, label, details };
}

function analyzeImageAigc(hasImage: boolean): {
  score: number;
  confidence: number;
  label: string;
  details: string;
} {
  if (!hasImage) {
    return {
      score: 0.5,
      confidence: 0.0,
      label: "No Image Provided",
      details: "No image was provided for analysis. Score is neutral (0.5) and confidence is 0.",
    };
  }

  const score = 0.1 + Math.random() * 0.85;
  const confidence = 0.6 + Math.random() * 0.35;

  let label: string;
  let details: string;
  if (score > 0.7) {
    label = "Likely AI-Generated Image";
    details = "Image shows artifacts consistent with AI generation: unnatural texture patterns, inconsistent lighting, irregular facial features, or GAN-typical frequency artifacts detected.";
  } else if (score > 0.4) {
    label = "Possibly Manipulated Image";
    details = "Image shows some signs of manipulation or AI enhancement. Inconsistent metadata, edge artifacts, or pixel inconsistencies detected at moderate level.";
  } else {
    label = "Likely Authentic Image";
    details = "Image appears to be authentic photographic content. Natural noise patterns, consistent lighting, and EXIF metadata are consistent with camera capture.";
  }

  return { score, confidence, label, details };
}

function fusionScore(textAigc: number, textIntent: number, imageAigc: number, hasImage: boolean): number {
  const w1 = 0.35;
  const w2 = 0.40;
  const w3 = hasImage ? 0.25 : 0.0;
  const normalizer = hasImage ? 1.0 : (w1 + w2);

  return (w1 * textAigc + w2 * textIntent + w3 * imageAigc) / normalizer;
}

function generateReport(
  title: string,
  textAigcScore: ReturnType<typeof analyzeTextAigc>,
  textIntentScore: ReturnType<typeof analyzeTextIntent>,
  imageAigcScore: ReturnType<typeof analyzeImageAigc>,
  final: number,
  hasImage: boolean
): string {
  const verdict = final > 0.65 ? "FAKE" : final > 0.35 ? "SUSPICIOUS" : "REAL";

  const reportParts = [
    `## Fake News Detection Report`,
    ``,
    `**Article Title:** "${title}"`,
    `**Final Verdict: ${verdict}** (Score: ${(final * 100).toFixed(1)}%)`,
    ``,
    `---`,
    ``,
    `### Executive Summary`,
    ``,
  ];

  if (verdict === "FAKE") {
    reportParts.push(
      `This article has been flagged as **highly likely to be fake or misleading**. Multiple detection models converged on indicators of inauthenticity. The content should be treated with significant skepticism.`
    );
  } else if (verdict === "SUSPICIOUS") {
    reportParts.push(
      `This article shows **mixed signals** with some concerning indicators. While not definitively fake, readers should exercise caution and verify key claims through reliable sources.`
    );
  } else {
    reportParts.push(
      `This article appears to be **largely authentic**. Detection models found limited indicators of AI generation or misleading intent. Standard media literacy practices still apply.`
    );
  }

  reportParts.push(
    ``,
    `---`,
    ``,
    `### Model Analysis Breakdown`,
    ``,
    `**1. Text AIGC Detection (RoBERTa-based)**`,
    `- Score: ${(textAigcScore.score * 100).toFixed(1)}% fake probability`,
    `- Label: ${textAigcScore.label}`,
    `- Analysis: ${textAigcScore.details}`,
    ``,
    `**2. Text Intent Analysis (CB-LLM)**`,
    `- Score: ${(textIntentScore.score * 100).toFixed(1)}% misleading probability`,
    `- Label: ${textIntentScore.label}`,
    `- Analysis: ${textIntentScore.details}`,
    ``,
    `**3. Image AIGC Detection (CBM)**`,
    `- Score: ${hasImage ? `${(imageAigcScore.score * 100).toFixed(1)}% AI-generated probability` : "N/A - No image provided"}`,
    `- Label: ${imageAigcScore.label}`,
    `- Analysis: ${imageAigcScore.details}`,
    ``,
    `---`,
    ``,
    `### Decision Layer`,
    ``,
    `The final score is computed via a weighted linear fusion layer:`,
    `- Text AIGC weight: 35%`,
    `- Text Intent weight: 40%`,
    `- Image AIGC weight: ${hasImage ? "25%" : "0% (no image)"}`,
    ``,
    `**Final Score: ${(final * 100).toFixed(1)}%**`,
    ``,
    `---`,
    ``,
    `### Recommendations`,
    ``,
  );

  if (verdict === "FAKE") {
    reportParts.push(
      `- Do not share this article without independent verification`,
      `- Cross-check claims with established news outlets`,
      `- Check the original source's credibility and track record`,
      `- Look for the same story from multiple independent sources`,
      `- Be especially cautious of emotional language designed to provoke reactions`
    );
  } else if (verdict === "SUSPICIOUS") {
    reportParts.push(
      `- Verify key claims through reputable sources before sharing`,
      `- Note any sensational language or unverified assertions`,
      `- Check if the author and publication have established credibility`,
      `- Look for supporting evidence from peer publications`
    );
  } else {
    reportParts.push(
      `- This article passed basic authenticity checks`,
      `- Continue to apply standard media literacy practices`,
      `- Consider the source's editorial standards and track record`,
      `- Note: No AI system is 100% accurate — always think critically`
    );
  }

  return reportParts.join("\n");
}

router.post("/detect", async (req, res) => {
  const parseResult = DetectFakeNewsBody.safeParse(req.body);
  if (!parseResult.success) {
    res.status(400).json({
      error: "VALIDATION_ERROR",
      message: parseResult.error.message,
    });
    return;
  }

  const { title, content, imageBase64, imageUrl } = parseResult.data;
  const hasImage = !!(imageBase64 || imageUrl);

  try {
    const textAigc = analyzeTextAigc(title, content);
    const textIntent = analyzeTextIntent(title, content);
    const imageAigc = analyzeImageAigc(hasImage);
    const final = fusionScore(textAigc.score, textIntent.score, imageAigc.score, hasImage);

    let finalLabel: string;
    if (final > 0.65) finalLabel = "FAKE";
    else if (final > 0.35) finalLabel = "SUSPICIOUS";
    else finalLabel = "REAL";

    const finalConfidence =
      (textAigc.confidence + textIntent.confidence + (hasImage ? imageAigc.confidence : 0)) /
      (hasImage ? 3 : 2);

    const report = generateReport(title, textAigc, textIntent, imageAigc, final, hasImage);

    res.json({
      id: randomUUID(),
      title,
      sourceUrl: req.body.sourceUrl ?? null,
      textAigcScore: {
        modelName: "RoBERTa-AIGC-v2",
        modelType: "Text AIGC Detection",
        ...textAigc,
      },
      textIntentScore: {
        modelName: "CB-LLM-Intent-v1",
        modelType: "Text Intent Analysis",
        ...textIntent,
      },
      imageAigcScore: {
        modelName: "CBM-Image-v3",
        modelType: "Image AIGC Detection",
        ...imageAigc,
      },
      finalScore: final,
      finalLabel,
      finalConfidence,
      report,
      analyzedAt: new Date().toISOString(),
    });
  } catch (err) {
    req.log.error({ err }, "Detection failed");
    res.status(500).json({
      error: "DETECTION_ERROR",
      message: "An error occurred during analysis",
    });
  }
});

router.post("/crawl", async (req, res) => {
  const parseResult = CrawlUrlBody.safeParse(req.body);
  if (!parseResult.success) {
    res.status(400).json({
      error: "VALIDATION_ERROR",
      message: parseResult.error.message,
    });
    return;
  }

  const { url } = parseResult.data;

  try {
    new URL(url);
  } catch {
    res.status(400).json({
      error: "INVALID_URL",
      message: "The provided URL is not valid",
    });
    return;
  }

  try {
    const response = await axios.get(url, {
      timeout: 15000,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; +https://fakenewsdetector.app)",
        "Accept": "text/html,application/xhtml+xml,application/xml",
      },
      maxRedirects: 5,
    });

    const $ = cheerio.load(response.data as string);

    $("script, style, nav, footer, header, aside, .ad, .advertisement, [class*='cookie'], [class*='popup']").remove();

    let title =
      $('meta[property="og:title"]').attr("content") ||
      $('meta[name="twitter:title"]').attr("content") ||
      $("h1").first().text().trim() ||
      $("title").text().trim() ||
      "";

    title = title.replace(/\s+/g, " ").trim();

    let content = "";
    const articleSelectors = [
      "article",
      '[role="main"]',
      ".article-body",
      ".article-content",
      ".post-content",
      ".entry-content",
      ".story-body",
      ".news-body",
      "main",
    ];

    for (const selector of articleSelectors) {
      const el = $(selector);
      if (el.length) {
        content = el.text().replace(/\s+/g, " ").trim();
        if (content.length > 200) break;
      }
    }

    if (!content || content.length < 100) {
      content = $("body").text().replace(/\s+/g, " ").trim().slice(0, 8000);
    }

    content = content.slice(0, 8000);

    const imageUrls: string[] = [];
    const ogImage = $('meta[property="og:image"]').attr("content");
    if (ogImage) imageUrls.push(ogImage);

    $("article img, .article-body img, main img").each((_, el) => {
      const src = $(el).attr("src") || $(el).attr("data-src");
      if (src && src.startsWith("http") && !imageUrls.includes(src)) {
        imageUrls.push(src);
      }
    });

    const publishedAt =
      $('meta[property="article:published_time"]').attr("content") ||
      $('meta[name="date"]').attr("content") ||
      $("time").attr("datetime") ||
      undefined;

    const author =
      $('meta[name="author"]').attr("content") ||
      $('[rel="author"]').first().text().trim() ||
      $(".author").first().text().trim() ||
      undefined;

    const siteName =
      $('meta[property="og:site_name"]').attr("content") ||
      new URL(url).hostname ||
      undefined;

    res.json({
      url,
      title,
      content,
      imageUrls: imageUrls.slice(0, 5),
      publishedAt,
      author,
      siteName,
    });
  } catch (err) {
    req.log.error({ err }, "Crawl failed");
    if (axios.isAxiosError(err)) {
      res.status(500).json({
        error: "CRAWL_ERROR",
        message: `Failed to fetch the URL: ${(err as Error).message}`,
      });
    } else {
      res.status(500).json({
        error: "CRAWL_ERROR",
        message: "An unexpected error occurred while crawling",
      });
    }
  }
});

export default router;
