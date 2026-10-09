/**
 * 簡單的 Markdown 文本渲染器
 * 支持常用的 markdown 格式：**粗體**、*斜體*、換行等
 */

interface MarkdownTextProps {
  content: string;
  className?: string;
}

export function MarkdownText({ content, className = '' }: MarkdownTextProps) {
  const renderInline = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
        return <strong key={idx} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
      }
      return <span key={idx}>{part}</span>;
    });
  };

  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];
  let pendingList: string[] = [];

  const flushList = () => {
    if (pendingList.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="list-disc pl-6 space-y-1 text-white/90">
        {pendingList.map((item, idx) => (
          <li key={idx}>{renderInline(item)}</li>
        ))}
      </ul>
    );
    pendingList = [];
  };

  lines.forEach((raw, idx) => {
    const line = raw.trim();

    if (!line) {
      flushList();
      blocks.push(<div key={`sp-${idx}`} className="h-2" />);
      return;
    }

    if (line.startsWith('- ')) {
      pendingList.push(line.slice(2));
      return;
    }

    flushList();

    if (line.startsWith('### ')) {
      blocks.push(
        <h3 key={`h3-${idx}`} className="text-lg font-semibold text-white mt-2 mb-1">
          {renderInline(line.slice(4))}
        </h3>
      );
      return;
    }

    if (line.startsWith('## ')) {
      blocks.push(
        <h2 key={`h2-${idx}`} className="text-xl font-semibold text-white mt-2 mb-1">
          {renderInline(line.slice(3))}
        </h2>
      );
      return;
    }

    if (line.startsWith('# ')) {
      blocks.push(
        <h1 key={`h1-${idx}`} className="text-2xl font-bold text-white mt-2 mb-1">
          {renderInline(line.slice(2))}
        </h1>
      );
      return;
    }

    blocks.push(
      <p key={`p-${idx}`} className="text-white/90 leading-relaxed">
        {renderInline(line)}
      </p>
    );
  });

  flushList();

  return (
    <div className={`${className} space-y-2`}>{blocks}</div>
  );
}
