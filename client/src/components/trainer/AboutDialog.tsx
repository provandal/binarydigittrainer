import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";

export interface AboutDialogProps {
  isAboutOpen: boolean;
  setIsAboutOpen: (v: boolean) => void;
}

export function AboutDialog({ isAboutOpen, setIsAboutOpen }: AboutDialogProps) {
  return (
    <Dialog open={isAboutOpen} onOpenChange={setIsAboutOpen}>
      <DialogContent className="max-w-[95vw] sm:max-w-md">
        <div className="space-y-4 text-center">
          <h2 className="text-xl font-bold">About Binary Digit Trainer</h2>

          <div className="space-y-3 text-sm text-gray-600">
            <p>
              An educational neural network training platform for binary digit recognition,
              featuring comprehensive step-by-step visualization and interactive learning tools.
            </p>

            <div className="border-t pt-3">
              <h3 className="mb-2 font-medium text-gray-800">Created by</h3>
              <div className="space-y-1">
                <div>
                  <strong>Erik Smith</strong>
                  <br />
                  <a href="mailto:erik.smith@dell.com" className="text-blue-600 hover:underline">
                    erik.smith@dell.com
                  </a>
                </div>
              </div>
            </div>

            <div className="border-t pt-3">
              <h3 className="mb-2 font-medium text-gray-800">Co-created with</h3>
              <div>
                <strong>Replit Agent</strong>
                <br />
                <span className="text-xs">
                  AI-powered development assistant providing implementation, architecture design,
                  and educational content creation
                </span>
              </div>
            </div>
          </div>

          <Button onClick={() => setIsAboutOpen(false)} className="mt-4 w-full">
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
