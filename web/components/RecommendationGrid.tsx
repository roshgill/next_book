"use client";

import type { Book } from "@/lib/api";
import BookCard from "./BookCard";

interface Props {
  books: Book[];
  isLoading: boolean;
}

function SkeletonCard() {
  return (
    <div className="flex flex-col bg-cream rounded-sm shadow-book overflow-hidden">
      <div className="skeleton w-full" style={{ aspectRatio: "2/3" }} />
      <div className="p-3 flex flex-col gap-2">
        <div className="skeleton h-4 rounded w-4/5" />
        <div className="skeleton h-3 rounded w-3/5" />
        <div className="skeleton h-3 rounded w-2/5 mt-1" />
      </div>
    </div>
  );
}

export default function RecommendationGrid({ books, isLoading }: Props) {
  if (isLoading) {
    return (
      <div>
        <p className="font-sans text-xs uppercase tracking-widest text-ink-light opacity-50 mb-4">
          Finding recommendations…
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (books.length === 0) {
    return (
      <div className="py-16 text-center">
        <p className="font-body italic text-ink-light opacity-60 text-lg">
          No recommendations yet — search for a book above.
        </p>
      </div>
    );
  }

  return (
    <div>
      <p className="font-sans text-xs uppercase tracking-widest text-ink-light opacity-50 mb-4">
        {books.length} recommendations
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {books.map((book, i) => (
          <div key={`${book.isbn13}-${i}`} className="stagger-item">
            <BookCard book={book} size="small" rank={i + 1} />
          </div>
        ))}
      </div>
    </div>
  );
}
