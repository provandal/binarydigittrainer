import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import BinaryDigitTrainer from "@/components/binary-digit-trainer";

function App() {
  return (
    <TooltipProvider>
      <Toaster />
      <BinaryDigitTrainer />
    </TooltipProvider>
  );
}

export default App;
