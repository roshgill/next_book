"use client";

type ModelName = "naive" | "classical" | "deep";

interface Props {
  selected: ModelName;
  onChange: (model: ModelName) => void;
  disabled?: boolean;
}

const MODELS: {
  name: ModelName;
  label: string;
  description: string;
}[] = [
  {
    name: "naive",
    label: "Popular",
    description: "Returns the most-rated books regardless of your choice",
  },
  {
    name: "classical",
    label: "TF-IDF",
    description: "Lexical word-overlap similarity across descriptions",
  },
  {
    name: "deep",
    label: "Semantic",
    description: "MiniLM embeddings + learned re-ranker (best results)",
  },
];

export default function ModelSelector({ selected, onChange, disabled }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-sans text-xs uppercase tracking-widest text-ink-light opacity-60">
        Recommendation model
      </p>
      <div className="inline-flex border border-parchment-dark rounded-sm overflow-hidden shadow-book">
        {MODELS.map((m, i) => {
          const isSelected = selected === m.name;
          return (
            <button
              key={m.name}
              onClick={() => !disabled && onChange(m.name)}
              disabled={disabled}
              title={m.description}
              className={`
                relative group px-5 py-2.5
                font-sans text-sm font-700 tracking-wide
                transition-all duration-150
                ${i > 0 ? "border-l border-parchment-dark" : ""}
                ${
                  isSelected
                    ? "bg-burgundy text-cream"
                    : "bg-cream text-ink-light hover:bg-parchment hover:text-ink"
                }
                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
              `}
            >
              {m.label}
              {/* Tooltip */}
              <span
                className="
                  pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                  w-48 px-3 py-2 rounded-sm
                  bg-ink text-cream text-xs font-body font-400 text-center leading-snug
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-150
                  shadow-book-hover z-50
                "
              >
                {m.description}
                <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-ink" />
              </span>
            </button>
          );
        })}
      </div>
      <p className="font-body italic text-sm text-ink-light opacity-60">
        {MODELS.find((m) => m.name === selected)?.description}
      </p>
    </div>
  );
}
