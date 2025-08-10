import { HelpCircle } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { MINI_TUTORIALS } from "@/lib/tutorials";

export function HelpIcon({ k }: { k: keyof typeof MINI_TUTORIALS }) {
  const t = MINI_TUTORIALS[k];
  if (!t) return null;
  
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button 
          aria-label={`Help: ${t.title}`} 
          className="ml-1 text-blue-600 hover:text-blue-700 align-middle inline-flex items-center justify-center"
        >
          <HelpCircle className="w-4 h-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="max-w-[34rem] prose prose-sm">
        <h4 className="font-semibold mb-2">{t.title}</h4>
        {/* We store HTML to allow subscript/superscript and italics in formulas */}
        <div dangerouslySetInnerHTML={{ __html: t.html }} />
      </PopoverContent>
    </Popover>
  );
}