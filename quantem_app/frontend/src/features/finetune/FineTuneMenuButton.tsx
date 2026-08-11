/**
 * The library header's way in. Owner R13's second entry point.
 *
 * Fetches nothing until it is pressed: this button sits on the first screen of
 * the application, and a screen that issues a request for a dialog nobody has
 * opened is a screen that is slower for everybody who never opens it. Which
 * organelle to train is the dialog's first question, because from here there is
 * no organelle in context to answer it.
 */

import { useState } from "react";
import { Button } from "@/shared/ui/design";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";

export function FineTuneMenuButton() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button onClick={() => setOpen(true)}>Fine-tune a model</Button>
      <FineTuneDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}
