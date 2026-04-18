"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getRecommendations, type Book } from "@/lib/api";
import SearchBar from "@/components/SearchBar";
import BookCard from "@/components/BookCard";
import ModelSelector from "@/components/ModelSelector";
import RecommendationGrid from "@/components/RecommendationGrid";

type ModelName = "naive" | "classical" | "deep";

const EXAMPLE_QUERIES: { isbn: string; title: string }[] = [
  { isbn: "9780002005883", title: "Gilead" },

  { isbn: "9780739360385", title: "Harry Potter and the Sorcerer's Stone" },
  { isbn: "9780618002214", title: "The Hobbit" },
];

export default function Home() {
  const [selectedIsbn, setSelectedIsbn] = useState<string | null>(null);
  const [queryBook, setQueryBook] = useState<Book | null>(null);
  const [recommendations, setRecommendations] = useState<Book[]>([]);
  const [model, setModel] = useState<ModelName>("deep");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const fetchRecs = useCallback(
    async (isbn: string, modelName: ModelName) => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getRecommendations(isbn, modelName, 10);
        setQueryBook(data.query);
        setRecommendations(data.recommendations);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
        setRecommendations([]);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  // Fetch when isbn or model changes
  useEffect(() => {
    if (selectedIsbn) {
      fetchRecs(selectedIsbn, model);
    }
  }, [selectedIsbn, model, fetchRecs]);

  // Scroll into view when results first appear
  useEffect(() => {
    if (queryBook && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [queryBook?.isbn13]);

  const handleSelect = (isbn: string) => {
    setSelectedIsbn(isbn);
  };

  const handleModelChange = (newModel: ModelName) => {
    setModel(newModel);
    // Clear current recs so the grid shows skeleton immediately
    setRecommendations([]);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="pt-16 pb-10 px-6 text-center">
        <div className="max-w-3xl mx-auto">
          <div className="inline-block mb-4">
            {/* Decorative rule */}
            <div className="flex items-center gap-3 justify-center mb-3 opacity-40">
              <div className="h-px w-12 bg-ink" />
              <svg width="12" height="12" viewBox="0 0 12 12" fill="#1a1108">
                <polygon points="6,0 7.5,4.5 12,4.5 8.5,7.5 10,12 6,9 2,12 3.5,7.5 0,4.5 4.5,4.5" />
              </svg>
              <div className="h-px w-12 bg-ink" />
            </div>

            <h1 className="font-serif text-5xl md:text-6xl text-ink tracking-tight leading-none">
              Next Book
            </h1>

            <div className="flex items-center gap-3 justify-center mt-3 opacity-40">
              <div className="h-px w-12 bg-ink" />
              <svg width="12" height="12" viewBox="0 0 12 12" fill="#1a1108">
                <polygon points="6,0 7.5,4.5 12,4.5 8.5,7.5 10,12 6,9 2,12 3.5,7.5 0,4.5 4.5,4.5" />
              </svg>
              <div className="h-px w-12 bg-ink" />
            </div>
          </div>

          <p className="font-body italic text-ink-light text-lg md:text-xl leading-relaxed opacity-80">
            Tell us a book you loved. We&rsquo;ll find what to read next.
          </p>
        </div>
      </header>

      {/* Search */}
      <section className="px-6 pb-8">
        <SearchBar onSelect={handleSelect} />

        {/* Example queries */}
        {!selectedIsbn && (
          <div className="flex flex-wrap gap-2 justify-center mt-5 animate-fade-up">
            <span className="font-sans text-xs uppercase tracking-widest text-ink-light opacity-50 self-center mr-1">
              Try:
            </span>
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q.isbn}
                onClick={() => handleSelect(q.isbn)}
                className="
                  px-3 py-1.5 rounded-sm
                  bg-cream border border-parchment-dark
                  font-body italic text-sm text-ink-light
                  hover:border-burgundy hover:text-burgundy
                  transition-colors duration-150 shadow-book
                "
              >
                {q.title}
              </button>
            ))}
          </div>
        )}
      </section>

      {/* Error */}
      {error && (
        <div className="px-6 mb-6 max-w-2xl mx-auto w-full">
          <div className="bg-rose-50 border border-rose-200 rounded-sm px-4 py-3 text-rose-800 font-body text-sm">
            {error}
          </div>
        </div>
      )}

      {/* Results */}
      {(queryBook || isLoading) && (
        <section
          ref={resultsRef}
          className="flex-1 px-6 pb-16 max-w-7xl mx-auto w-full animate-fade-up"
        >
          {/* Divider */}
          <div className="flex items-center gap-4 mb-8 opacity-30">
            <div className="flex-1 h-px bg-ink" />
            <svg width="10" height="10" viewBox="0 0 10 10" fill="#1a1108">
              <polygon points="5,0 6.5,3.5 10,3.5 7,5.5 8.5,10 5,7.5 1.5,10 3,5.5 0,3.5 3.5,3.5" />
            </svg>
            <div className="flex-1 h-px bg-ink" />
          </div>

          {/* Query book card */}
          {queryBook && (
            <div className="mb-8">
              <BookCard book={queryBook} size="large" />
            </div>
          )}

          {/* Model selector */}
          <div className="mb-8">
            <ModelSelector
              selected={model}
              onChange={handleModelChange}
              disabled={isLoading}
            />
          </div>

          {/* Recommendations */}
          <RecommendationGrid books={recommendations} isLoading={isLoading} />
        </section>
      )}

      {/* Footer */}
      <footer className="mt-auto py-8 px-6 border-t border-parchment-dark text-center">
        <p className="font-body italic text-xs text-ink-light opacity-40">
          Content-based recommendation &mdash; 5,787 books &mdash; three models
        </p>
      </footer>
    </div>
  );
}
