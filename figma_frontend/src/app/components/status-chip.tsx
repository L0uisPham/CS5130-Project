import { Badge } from "./ui/badge";

type Status = "present" | "uncertain" | "not-present";

interface StatusChipProps {
  status: Status;
}

export function StatusChip({ status }: StatusChipProps) {
  const config = {
    present: {
      label: "Present",
      className: "bg-green-100 text-green-800 hover:bg-green-100 text-xs px-2 py-0",
    },
    uncertain: {
      label: "Uncertain",
      className: "bg-amber-100 text-amber-800 hover:bg-amber-100 text-xs px-2 py-0",
    },
    "not-present": {
      label: "Not present",
      className: "bg-gray-100 text-gray-600 hover:bg-gray-100 text-xs px-2 py-0",
    },
  };

  const { label, className } = config[status];

  return (
    <Badge variant="secondary" className={className}>
      {label}
    </Badge>
  );
}