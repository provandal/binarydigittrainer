import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { X, ArrowLeft, ArrowRight } from 'lucide-react';

export interface TourStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for highlighting
  position?: 'top' | 'bottom' | 'left' | 'right';
  validation?: () => boolean; // Function to check if user completed the required action
  action?: string; // Description of required action
  waitForAction?: boolean; // Whether to wait for user action before enabling Next
}

interface GuidedTourProps {
  isOpen: boolean;
  onClose: () => void;
  onReset?: () => void; // Reset network function
  tourSteps: TourStep[];
}

export default function GuidedTour({ isOpen, onClose, onReset, tourSteps }: GuidedTourProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [highlightedElement, setHighlightedElement] = useState<Element | null>(null);
  const [validationPassed, setValidationPassed] = useState(false);

  // Reset tour state when opened
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      setValidationPassed(false);
      // Reset network if function provided
      if (onReset) {
        onReset();
      }
    }
  }, [isOpen, onReset]);

  // Update highlighting when step changes
  useEffect(() => {
    if (!isOpen) return;

    const step = tourSteps[currentStep];
    if (step?.target) {
      const element = document.querySelector(step.target);
      if (element) {
        setHighlightedElement(element);
        // Add highlight class
        element.classList.add('tour-highlight');
        // Scroll into view
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } else {
      setHighlightedElement(null);
    }

    // Cleanup previous highlights
    return () => {
      document.querySelectorAll('.tour-highlight').forEach(el => {
        el.classList.remove('tour-highlight');
      });
    };
  }, [currentStep, isOpen, tourSteps]);

  // Validation checking
  useEffect(() => {
    if (!isOpen) return;

    const step = tourSteps[currentStep];
    if (step?.validation && step.waitForAction) {
      const checkValidation = () => {
        const passed = step.validation!();
        setValidationPassed(passed);
      };

      // Check immediately
      checkValidation();

      // Set up interval to check periodically
      const interval = setInterval(checkValidation, 500);
      return () => clearInterval(interval);
    } else {
      setValidationPassed(true);
    }
  }, [currentStep, isOpen, tourSteps]);

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1);
      setValidationPassed(false);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setValidationPassed(false);
    }
  };

  const handleClose = () => {
    // Clean up highlights
    document.querySelectorAll('.tour-highlight').forEach(el => {
      el.classList.remove('tour-highlight');
    });
    onClose();
  };

  if (!isOpen) return null;

  const step = tourSteps[currentStep];
  const isLastStep = currentStep === tourSteps.length - 1;
  const canProceed = !step?.waitForAction || validationPassed;

  return (
    <>
      {/* Overlay for highlighting */}
      <div className="fixed inset-0 bg-black bg-opacity-50 z-40 pointer-events-none" />
      
      {/* Tour Dialog */}
      <Dialog open={isOpen} onOpenChange={() => {}} modal={false}>
        <DialogContent className="max-w-md z-50 fixed top-16 right-4 max-h-[calc(100vh-8rem)] overflow-y-auto shadow-lg border bg-white"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
        >
          <DialogTitle className="sr-only">Guided Tour Step {currentStep + 1}</DialogTitle>
          <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between tour-dialog-header">
              <div className="flex items-center space-x-2">
                <h3 className="font-semibold">Guided Tour</h3>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                  Step {currentStep + 1} of {tourSteps.length}
                </span>
              </div>
              <Button variant="ghost" size="sm" onClick={handleClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h4 className="font-medium text-sm">{step.title}</h4>
              
              <div className="text-sm text-gray-600 space-y-2">
                <div dangerouslySetInnerHTML={{ __html: step.content }} />
                
                {step.action && step.waitForAction && (
                  <div className="bg-blue-50 border border-blue-200 rounded p-2 mt-3">
                    <div className="text-xs font-medium text-blue-700 mb-1">
                      Action Required:
                    </div>
                    <div className="text-xs text-blue-600">
                      {step.action}
                    </div>
                    {!validationPassed && (
                      <div className="text-xs text-orange-600 mt-1">
                        ⏳ Waiting for you to complete this action...
                      </div>
                    )}
                    {validationPassed && (
                      <div className="text-xs text-green-600 mt-1">
                        ✓ Action completed! You can proceed.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Navigation */}
            <div className="flex justify-between pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePrevious}
                disabled={currentStep === 0}
              >
                <ArrowLeft className="h-3 w-3 mr-1" />
                Previous
              </Button>

              {isLastStep ? (
                <Button size="sm" onClick={handleClose}>
                  Finish Tour
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={handleNext}
                  disabled={!canProceed}
                >
                  Next
                  <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Inject CSS for highlighting and dialog positioning */}
      <style>{`
        .tour-highlight {
          position: relative;
          z-index: 45 !important;
          box-shadow: 0 0 0 2px #3B82F6, 0 0 0 4px rgba(59, 130, 246, 0.3) !important;
          border-radius: 4px !important;
        }
        
        /* Ensure dialog stays within viewport */
        [data-radix-popper-content-wrapper] {
          transform: none !important;
          position: fixed !important;
          top: 4rem !important;
          right: 1rem !important;
          left: auto !important;
          bottom: auto !important;
          max-width: calc(100vw - 2rem) !important;
          max-height: calc(100vh - 5rem) !important;
        }
        
        /* Better dialog positioning */
        [role="dialog"] {
          position: fixed !important;
          top: 4rem !important;
          right: 1rem !important;
          left: auto !important;
          bottom: auto !important;
          margin: 0 !important;
          transform: none !important;
        }
        
        /* Make dialog draggable by header */
        .tour-dialog-header {
          cursor: move;
          user-select: none;
        }
        
        /* Hide the automatic close button from DialogContent */
        [role="dialog"] button[aria-label="Close"] {
          display: none !important;
        }
        
        /* Alternative selector for close button */
        [data-radix-collection-item] {
          display: none !important;
        }
      `}</style>
    </>
  );
}