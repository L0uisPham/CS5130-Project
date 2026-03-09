import { StatusChip } from "./status-chip";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { Info } from "lucide-react";

interface ProbabilityRowProps {
  label: string;
  probability: number;
  status: "present" | "uncertain" | "not-present";
}

export function ProbabilityRow({
  label,
  probability,
  status,
}: ProbabilityRowProps) {
  const percentage = (probability * 100).toFixed(1);

  return (
    <div className="flex items-center gap-2 py-2 border-b last:border-b-0">
      <div className="flex-1 min-w-0">
        <span className="text-xs block truncate">{label}</span>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${
              status === "present"
                ? "bg-green-500"
                : status === "uncertain"
                ? "bg-amber-500"
                : "bg-gray-400"
            }`}
            style={{ width: `${probability * 100}%` }}
          />
        </div>
        <span className="text-xs w-9 text-right tabular-nums">{percentage}%</span>
        <div className="w-20">
          <StatusChip status={status} />
        </div>
      </div>
    </div>
  );
}