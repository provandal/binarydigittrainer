import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { X, ArrowLeft, ArrowRight, Minimize2, Maximize2 } from 'lucide-react';

export interface TourStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for highlighting
  position?: 'top' | 'bottom' | 'left' | 'right';
  validation?: () => boolean; // Function to check if user completed the required action
  action?: string; // Description of required action
  waitForAction?: boolean; // Whether to wait for user action before enabling Next
  autoAdvanceOnValid?: boolean; // If true, advance automatically when validation turns true
  pin?: 'bottom-left' | 'top-right'; // Simple presets for dialog placement
  onNext?: () => void; // Callback to execute when Next button is clicked
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
  const [minimized, setMinimized] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Helper: does this step require validation to proceed?
  const requiresValidation = (step?: TourStep) =>
    !!step?.validation || !!step?.waitForAction;

  // Validation trigger function that can be called from outside
  const triggerValidation = () => {
    // Use a function to get current step and tour steps to avoid stale closure
    setCurrentStep(current => {
      const step = tourSteps[current];
      console.log('🔍 TOUR: triggerValidation called for step', current, 'stepId:', step?.id);
      if (!step) return current;
      let isValid = false;
      if (step.validation) {
        isValid = !!step.validation();
        console.log('🔍 TOUR: validation() →', isValid);
      } else if (requiresValidation(step)) {
        // No validation function, but this step still requires an external acknowledgement.
        // Calling triggerValidation() marks it complete.
        isValid = true;
        console.log('🔍 TOUR: no validation(), treating external trigger as completion');
      }
      if (requiresValidation(step)) {
        setValidationPassed(isValid);
        if (isValid && step.autoAdvanceOnValid) {
          setTimeout(() => {
            setCurrentStep(c => Math.min(c + 1, tourSteps.length - 1));
            setValidationPassed(false);
          }, 0);
        }
      }
      return current; // Don't change the step
    });
  };

  // Reset tour state when opened and provide validation trigger to parent
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      setValidationPassed(false);
      setMinimized(false);
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
      // Default (top-right) or pinned preset
      if (step?.pin === 'bottom-left') {
        setDialogPosition({ x: 16, y: window.innerHeight - 400 });
      } else {
        setDialogPosition({ x: 0, y: 0 }); // default top-right
      }
    }
  }, [currentStep, isOpen]);

  // Mouse drag handlers
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

  // Touch drag handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    if ((e.target as HTMLElement).closest('.tour-dialog-header')) {
      const touch = e.touches[0];
      const rect = dialogRef.current?.getBoundingClientRect();
      if (rect) {
        setIsDragging(true);
        setDragOffset({
          x: touch.clientX - rect.left,
          y: touch.clientY - rect.top
        });
      }
    }
  };

  useEffect(() => {
    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging) {
        e.preventDefault();
        const touch = e.touches[0];
        setDialogPosition({
          x: touch.clientX - dragOffset.x,
          y: touch.clientY - dragOffset.y
        });
      }
    };

    const handleTouchEnd = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('touchmove', handleTouchMove, { passive: false });
      document.addEventListener('touchend', handleTouchEnd);
    }

    return () => {
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleTouchEnd);
    };
  }, [isDragging, dragOffset]);

  // Update highlighting when step changes
  useEffect(() => {
    if (!isOpen) return;

    const step = tourSteps[currentStep];
    // Cleanup previous highlights
    document.querySelectorAll('.tour-highlight').forEach(el => {
      el.classList.remove('tour-highlight');
    });

    if (step?.target) {
      // Support multiple targets separated by commas
      const targets = step.target.split(',').map(t => t.trim());
      let firstElement: Element | null = null;

      targets.forEach(target => {
        const element = document.querySelector(target);
        if (element) {
          if (!firstElement) firstElement = element;
          element.classList.add('tour-highlight');
        }
      });

      if (firstElement) {
        setHighlightedElement(firstElement);
        // Scroll into view using the first element
        (firstElement as Element).scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } else {
      setHighlightedElement(null);
    }

    // Cleanup on component unmount
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
    // New logic: if the step REQUIRES validation (either it has validation() OR waitForAction),
    // start with "not passed" and wait for triggerValidation() or a true validation().
    if (requiresValidation(step)) {
      console.log('🔍 TOUR: Step requires validation → validationPassed = false');
      setValidationPassed(false);
    } else {
      console.log('🔍 TOUR: Step does not require validation → validationPassed = true');
      setValidationPassed(true);
    }
  }, [currentStep, isOpen]); // Remove tourSteps from dependencies to prevent re-render loop

  // NEW: actively re-evaluate validation() for steps that define it.
  // This keeps the "Next" button disabled until validation turns true.
  useEffect(() => {
    if (!isOpen) return;

    const step = tourSteps[currentStep];
    if (!step?.validation) return; // only poll for steps that define validation()

    let id: number | null = null;
    let lastValidationResult = false;

    const tick = () => {
      // Guard against exceptions inside validation()
      let ok = false;
      try {
        ok = !!step.validation?.();
      } catch (e) {
        ok = false;
      }

      // Only update if validation result actually changed (prevents UI jitter)
      if (ok !== lastValidationResult) {
        lastValidationResult = ok;
        setValidationPassed(ok);
      }

      if (ok && step.autoAdvanceOnValid) {
        // Advance once, then stop polling
        setCurrentStep(c => Math.min(c + 1, tourSteps.length - 1));
        return;
      }

      // Schedule next check with longer interval to reduce flicker
      id = window.setTimeout(tick, 500);
    };

    tick();

    return () => {
      if (id !== null) window.clearTimeout(id);
    };
  }, [currentStep, isOpen, tourSteps]);

  const handleNext = () => {
    const step = tourSteps[currentStep];

    // Call onNext callback if defined (e.g., auto-stop training)
    if (step?.onNext) {
      step.onNext();
    }

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
  // Only allow Next if either the step does not require validation, or it has passed.
  const canProceed = !requiresValidation(step) || validationPassed;

  // Calculate dialog style based on position
  const dialogStyle = (() => {
    // Center first two steps (welcome and reset)
    if ((currentStep === 0 || currentStep === 1) && dialogPosition.x === 0 && dialogPosition.y === 0) {
      return {
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)'
      };
    }
    // Adjust position for steps 5 and 6 (index 4 and 5) to prevent cutoff
    if ((currentStep === 4 || currentStep === 5) && dialogPosition.x === 0 && dialogPosition.y === 0) {
      return {
        bottom: '1rem',
        left: '1rem',
        maxHeight: 'calc(100vh - 2rem)',
        overflow: 'auto' as const
      };
    }
    // Position final step (16 of 16) to the right of weight box
    if (currentStep === 15 && dialogPosition.x === 0 && dialogPosition.y === 0) {
      return {
        top: '50%',
        left: '60%',
        transform: 'translateY(-50%)'
      };
    }
    // Default positioning for other steps
    return dialogPosition.x === 0 && dialogPosition.y === 0
      ? { top: '4rem', right: '1rem' } // Default top-right position
      : { left: dialogPosition.x, top: dialogPosition.y }; // Custom dragged position
  })();

  return (
    <>
      {/* Overlay for highlighting — tap-through on mobile when minimized */}
      <div className={`fixed inset-0 bg-black z-40 pointer-events-none ${minimized ? 'bg-opacity-20' : 'bg-opacity-50'}`} />

      {/* Minimized pill — mobile only */}
      {minimized && (
        <div
          className="fixed z-50 bottom-3 left-1/2 -translate-x-1/2"
        >
          <button
            onClick={() => setMinimized(false)}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-full shadow-lg active:bg-blue-700 transition-colors"
          >
            <Maximize2 className="h-4 w-4" />
            <span className="text-sm font-medium">
              Tour {currentStep + 1}/{tourSteps.length}
            </span>
            {step.waitForAction && !validationPassed && (
              <span className="text-xs bg-blue-500 px-1.5 py-0.5 rounded-full">action needed</span>
            )}
          </button>
        </div>
      )}

      {/* Full Tour Modal */}
      {!minimized && (
        <div
          ref={dialogRef}
          className="fixed max-w-[calc(100vw-1rem)] sm:max-w-md w-full z-50"
          style={dialogStyle}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
        >
          <div className="bg-white rounded-lg shadow-xl border max-h-[calc(100vh-8rem)] overflow-hidden">
            <div className="overflow-y-auto max-h-[calc(100vh-8rem)]">
              <div className="space-y-3 sm:space-y-4 p-3 sm:p-4">
                {/* Header */}
                <div className="flex items-center justify-between tour-dialog-header cursor-move select-none">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-sm sm:text-base">Guided Tour</h3>
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 sm:py-1 rounded">
                      {currentStep + 1}/{tourSteps.length}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {/* Minimize button — visible on mobile */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setMinimized(true)}
                      className="sm:hidden h-8 w-8 p-0"
                      title="Minimize tour"
                    >
                      <Minimize2 className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={handleClose} className="h-8 w-8 p-0">
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* Content */}
                <div className="space-y-2 sm:space-y-3">
                  <h4 className="font-medium text-xs sm:text-sm">{step.title}</h4>

                  <div className="text-xs sm:text-sm text-gray-600 space-y-2">
                    <div dangerouslySetInnerHTML={{ __html: step.content }} />

                    {step.action && step.waitForAction && (
                      <div className="bg-blue-50 border border-blue-200 rounded p-2 mt-2 sm:mt-3">
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
                <div className="flex justify-between pt-1 sm:pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handlePrevious}
                    disabled={currentStep === 0}
                    className="text-xs sm:text-sm"
                  >
                    <ArrowLeft className="h-3 w-3 mr-1" />
                    Prev
                  </Button>

                  {isLastStep ? (
                    <Button size="sm" onClick={handleClose} className="text-xs sm:text-sm">
                      Finish Tour
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={handleNext}
                      disabled={!canProceed}
                      className="text-xs sm:text-sm"
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
      )}

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
