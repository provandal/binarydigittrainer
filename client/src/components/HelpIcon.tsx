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
          className="ml-1 inline-flex items-center justify-center align-middle text-blue-600 hover:text-blue-700"
        >
          <HelpCircle className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="prose prose-sm max-w-[34rem]">
        <h4 className="mb-2 font-semibold">{t.title}</h4>
        {/* We store HTML to allow subscript/superscript and italics in formulas */}
        <div dangerouslySetInnerHTML={{ __html: t.html }} />
      </PopoverContent>
    </Popover>
  );
}
