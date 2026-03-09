import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription } from "./ui/alert";

export function DisclaimerBanner() {
  return (
    <Alert className="border-amber-200 bg-amber-50">
      <AlertTriangle className="h-4 w-4 text-amber-600" />
      <AlertDescription className="text-amber-800">
        <strong>For research and education only.</strong> Not for clinical
        decision-making.
      </AlertDescription>
    </Alert>
  );
}
