import { AlertTriangle, Database, ShieldCheck } from "lucide-react";
import type { SimulationResponse } from "@/lib/types";

interface ResearchTransparencyProps {
  response: SimulationResponse;
}

function reported(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === ""
    ? "Not reported by service"
    : String(value);
}

export default function ResearchTransparency({ response }: ResearchTransparencyProps) {
  const metadata = [
    { label: "Data cutoff", value: reported(response.data_cutoff) },
    { label: "Model version", value: reported(response.model_version) },
    { label: "Prediction year", value: reported(response.prediction_year) },
    {
      label: "Uncertainty shown",
      value: response.interval_label || "90% prediction interval",
    },
  ];

  return (
    <section
      aria-labelledby="model-transparency-heading"
      className="overflow-hidden rounded-md border border-neutral-800 bg-neutral-900/30"
    >
      <div className="flex items-center gap-2 border-b border-neutral-800 px-4 py-3">
        <Database size={15} className="text-cyan-400" aria-hidden="true" />
        <h3 id="model-transparency-heading" className="text-sm font-semibold text-neutral-100">
          Model transparency
        </h3>
      </div>
      <dl className="grid grid-cols-2 border-b border-neutral-800 lg:grid-cols-4">
        {metadata.map((item) => (
          <div key={item.label} className="border-b border-r border-neutral-800 p-3 last:border-r-0 lg:border-b-0">
            <dt className="text-[11px] uppercase tracking-wide text-neutral-600">{item.label}</dt>
            <dd className="mt-1 text-xs font-medium text-neutral-300">{item.value}</dd>
          </div>
        ))}
      </dl>
      <div className="grid gap-4 p-4 lg:grid-cols-2">
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-neutral-300">
            <ShieldCheck size={14} className="text-emerald-400" aria-hidden="true" />
            Bucket meaning
          </p>
          <ul className="mt-2 space-y-1 text-xs leading-relaxed text-neutral-500">
            <li><strong className="text-emerald-300">Safe:</strong> rank ÷ predicted cutoff ≤ 0.80.</li>
            <li><strong className="text-amber-300">Target:</strong> ratio above 0.80 and ≤ 1.05.</li>
            <li><strong className="text-red-300">Reach:</strong> ratio above 1.05 and ≤ 1.40.</li>
          </ul>
        </div>
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-neutral-300">
            <AlertTriangle size={14} className="text-amber-400" aria-hidden="true" />
            Limitations
          </p>
          <p className="mt-2 text-xs leading-relaxed text-neutral-500">
            Historical cutoff patterns may not capture policy, seat-matrix, applicant-preference,
            or distribution shifts. Sparse programme histories can widen error. Confirm eligibility
            and current rules on the official JoSAA site.
          </p>
        </div>
      </div>
      <p className="border-t border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs font-medium text-amber-200">
        {response.decision_support_disclaimer ||
          "Decision support only; the uncalibrated admission-likelihood estimate is not an individual outcome probability or admission guarantee."}
      </p>
    </section>
  );
}
