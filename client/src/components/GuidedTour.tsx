import React, { useState, useEffect } from 'react';
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
  onValidationTrigger?: (triggerValidation: () => void) => void; // Callback to provide validation trigger
}

export default function GuidedTour({ isOpen, onClose, onReset, tourSteps, onValidationTrigger }: GuidedTourProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [highlightedElement, setHighlightedElement] = useState<Element | null>(null);
  const [validationPassed, setValidationPassed] = useState(false);
  const [dialogPosition, setDialogPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Validation trigger function that can be called from outside
  const triggerValidation = () => {
    // Use a function to get current step and tour steps to avoid stale closure
    setCurrentStep(current => {
      const step = tourSteps[current];
      console.log('🔍 TOUR: triggerValidation called for step', current, 'stepId:', step?.id);
      if (step?.validation) {
        const isValid = step.validation();
        console.log('🔍 TOUR: validation result:', isValid, 'setting validationPassed');
        setValidationPassed(isValid);
      } else {
        console.log('🔍 TOUR: no validation function for this step');
      }
      return current; // Don't change the step
    });
  };

  // Reset tour state when opened and provide validation trigger to parent
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      setValidationPassed(false);
      // Reset dialog position - default to top-right, except for step 5 (index 4)
      setDialogPosition({ x: 0, y: 0 });
      // Reset network if function provided
      if (onReset) {
        onReset();
      }
      // Provide validation trigger to parent component
      if (onValidationTrigger) {
        onValidationTrigger(triggerValidation);
      }
    }
  }, [isOpen]); // Remove onReset from dependencies to prevent infinite loop

  // Update dialog position when step changes
  useEffect(() => {
    if (isOpen) {
      const step = tourSteps[currentStep];
      // Position steps in lower left to avoid covering buttons
      if (step?.id === 'start-training' || step?.id === 'complete-training' || step?.id === 'dataset-training') {
        setDialogPosition({ x: 16, y: window.innerHeight - 400 });
      } else {
        // Default position (top-right)
        setDialogPosition({ x: 0, y: 0 });
      }
    }
  }, [currentStep, isOpen]);

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.tour-dialog-header')) {
      setIsDragging(true);
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setDialogPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

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

  // Validation checking - only for steps that don't require manual action
  useEffect(() => {
    if (!isOpen) return;

    const step = tourSteps[currentStep];
    console.log('🔍 TOUR: Step changed to', currentStep, 'stepId:', step?.id, 'waitForAction:', step?.waitForAction);
    if (step?.validation && step.waitForAction) {
      // For steps that wait for action, start with validation failed
      // Validation will be triggered manually when action occurs
      console.log('🔍 TOUR: Setting validationPassed = false (waiting for action)');
      setValidationPassed(false);
    } else {
      console.log('🔍 TOUR: Setting validationPassed = true (no wait required)');
      setValidationPassed(true);
    }
  }, [currentStep, isOpen]); // Remove tourSteps from dependencies to prevent re-render loop

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

  // Calculate dialog style based on position
  const dialogStyle = dialogPosition.x === 0 && dialogPosition.y === 0
    ? { top: '4rem', right: '1rem' } // Default top-right position
    : { left: dialogPosition.x, top: dialogPosition.y }; // Custom dragged position

  return (
    <>
      {/* Overlay for highlighting */}
      <div className="fixed inset-0 bg-black bg-opacity-50 z-40 pointer-events-none" />
      
      {/* Custom Tour Modal */}
      <div 
        className="fixed max-w-md w-full z-50" 
        style={dialogStyle}
        onMouseDown={handleMouseDown}
      >
        <div className="bg-white rounded-lg shadow-xl border max-h-[calc(100vh-8rem)] overflow-hidden">
          <div className="overflow-y-auto max-h-[calc(100vh-8rem)]">
            <div className="space-y-4 p-4">
              {/* Header */}
              <div className="flex items-center justify-between tour-dialog-header cursor-move select-none">
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
          </div>
        </div>
      </div>

      {/* Inject CSS for highlighting */}
      <style>{`
        .tour-highlight {
          position: relative;
          z-index: 45 !important;
          box-shadow: 0 0 0 2px #3B82F6, 0 0 0 4px rgba(59, 130, 246, 0.3) !important;
          border-radius: 4px !important;
        }
        
        /* Make dialog header draggable */
        .tour-dialog-header {
          cursor: move;
          user-select: none;
        }
      `}</style>
    </>
  );
}