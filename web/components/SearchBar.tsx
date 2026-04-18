"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { searchBooks, type SearchResult } from "@/lib/api";

interface Props {
  onSelect: (isbn: string, title: string) => void;
  placeholder?: string;
}

export default function SearchBar({
  onSelect,
  placeholder = "Search by title — try Gilead, The Hobbit, Dune…",
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [isLoading, setIsLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const runSearch = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }
    setIsLoading(true);
    try {
      const data = await searchBooks(q, 8);
      setResults(data);
      setIsOpen(data.length > 0);
      setActiveIndex(-1);
    } catch {
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runSearch(val), 300);
  };

  const handleSelect = (item: SearchResult) => {
    setQuery(item.title);
    setIsOpen(false);
    setResults([]);
    onSelect(item.isbn13, item.title);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, -1));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      handleSelect(results[activeIndex]);
    } else if (e.key === "Escape") {
      setIsOpen(false);
      setQuery("");
      setResults([]);
    }
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full max-w-2xl mx-auto">
      {/* Input */}
      <div className="relative">
        {/* Search icon */}
        <svg
          className="absolute left-4 top-1/2 -translate-y-1/2 text-ink-light opacity-50"
          width="18"
          height="18"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
        >
          <circle cx="8.5" cy="8.5" r="5.5" />
          <line x1="13" y1="13" x2="18" y2="18" />
        </svg>

        <input
          type="text"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setIsOpen(true)}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
          className="
            w-full pl-11 pr-4 py-4
            bg-cream border border-parchment-dark
            rounded-sm
            font-body text-[17px] text-ink placeholder:text-ink-light/50
            shadow-book
            focus:outline-none focus:border-burgundy focus:shadow-[0_0_0_3px_rgba(139,26,26,0.10)]
            transition-all duration-200
          "
        />

        {isLoading && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="w-4 h-4 rounded-full border-2 border-parchment-dark border-t-burgundy animate-spin" />
          </div>
        )}
      </div>

      {/* Dropdown */}
      {isOpen && results.length > 0 && (
        <ul
          className="
            absolute z-50 w-full mt-1
            bg-cream border border-parchment-dark
            rounded-sm shadow-book-hover
            overflow-hidden
            animate-fade-in
          "
          role="listbox"
        >
          {results.map((item, i) => (
            <li
              key={item.isbn13}
              role="option"
              aria-selected={i === activeIndex}
              onMouseDown={() => handleSelect(item)}
              onMouseEnter={() => setActiveIndex(i)}
              className={`
                px-4 py-3 cursor-pointer
                font-body text-[16px] text-ink
                border-b border-parchment-dark last:border-b-0
                transition-colors duration-100
                ${i === activeIndex ? "bg-parchment" : "hover:bg-parchment/60"}
              `}
            >
              {item.title}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
