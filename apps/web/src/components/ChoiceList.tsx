"use client";

import { X } from "lucide-react";
import type { PredictionResult } from "@/lib/types";
import { getTierForResult } from "@/lib/types";

interface ChoiceListProps {
  items: PredictionResult[];
  note: string;
  onNoteChange: (note: string) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

const tierLabel = {
  safe: "Safe",
  target: "Target",
  reach: "Reach",
};

export default function ChoiceList({
  items,
  note,
  onNoteChange,
  onRemove,
  onClear,
}: ChoiceListProps) {
  const counts = items.reduce(
    (acc, item) => {
      acc[getTierForResult(item)] += 1;
      return acc;
    },
    { safe: 0, target: 0, reach: 0 }
  );

  return (
    <aside className="lg:sticky lg:top-20 space-y-4">
      <div className="border border-neutral-800 rounded-md bg-neutral-900/35">
        <div className="flex items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
          <div>
            <h3 className="text-sm font-semibold text-white">Choice list</h3>
            <p className="text-xs text-neutral-500">
              {items.length ? `${items.length} saved options` : "Save rows while comparing"}
            </p>
          </div>
          {items.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              className="text-xs text-neutral-500 hover:text-neutral-200 transition-colors"
            >
              Clear
            </button>
          )}
        </div>

        <div className="grid grid-cols-3 border-b border-neutral-800 text-center">
          {(["safe", "target", "reach"] as const).map((tier) => (
            <div key={tier} className="px-3 py-3 border-r border-neutral-800 last:border-r-0">
              <p className="text-lg font-semibold text-white tabular-nums">{counts[tier]}</p>
              <p className="text-[11px] text-neutral-500">{tierLabel[tier]}</p>
            </div>
          ))}
        </div>

        <div className="max-h-[360px] overflow-y-auto">
          {items.length === 0 ? (
            <div className="px-4 py-8 text-sm text-neutral-500">
              Add a few options from the table to build a working JoSAA preference list.
            </div>
          ) : (
            <ol className="divide-y divide-neutral-800">
              {items.map((item, index) => (
                <li key={item.id} className="px-4 py-3">
                  <div className="flex items-start gap-3">
                    <span className="mt-0.5 w-6 shrink-0 text-xs text-neutral-500 tabular-nums">
                      {index + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-neutral-100">
                        {item.institute_name}
                      </p>
                      <p className="truncate text-xs text-neutral-500">{item.program_name}</p>
                      <div className="mt-2 flex items-center gap-2 text-[11px] text-neutral-500">
                        <span>{item.institute_type}</span>
                        <span>{item.quota_applied}</span>
                        <span className="text-neutral-300">
                          {item.probability_percent.toFixed(1)}% likelihood estimate
                        </span>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => onRemove(item.id)}
                      className="rounded-sm p-1 text-neutral-600 hover:bg-neutral-800 hover:text-neutral-200 transition-colors"
                      aria-label={`Remove ${item.institute_name}`}
                    >
                      <X size={14} />
                    </button>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="border border-neutral-800 rounded-md bg-neutral-900/35 p-4">
        <label className="block text-sm font-medium text-neutral-200" htmlFor="choice-notes">
          Counselling notes
        </label>
        <textarea
          id="choice-notes"
          value={note}
          onChange={(event) => onNoteChange(event.target.value)}
          rows={5}
          placeholder="Fees, distance from home, branch preference, backup logic..."
          className="mt-3 w-full resize-none rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 placeholder:text-neutral-600 outline-none transition-colors focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/25"
        />
      </div>
    </aside>
  );
}
