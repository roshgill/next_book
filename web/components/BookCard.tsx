"use client";

import { useState } from "react";
import type { Book } from "@/lib/api";

interface Props {
  book: Book;
  size: "large" | "small";
  rank?: number;
}

function CoverPlaceholder({ size }: { size: "large" | "small" }) {
  return (
    <div
      className={`
        flex-shrink-0 bg-parchment-dark flex items-center justify-center
        ${size === "large" ? "w-[140px] h-[210px]" : "w-[90px] h-[135px]"}
      `}
      style={{ borderRadius: "2px" }}
    >
      <svg
        width={size === "large" ? 36 : 24}
        height={size === "large" ? 36 : 24}
        viewBox="0 0 24 24"
        fill="none"
        stroke="#b89070"
        strokeWidth="1.5"
      >
        <path d="M4 19.5A2.5 2.5 0 016.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
      </svg>
    </div>
  );
}

function RatingBadge({ rating }: { rating: number }) {
  const color =
    rating >= 4.2
      ? "bg-emerald-50 text-emerald-800 border-emerald-200"
      : rating >= 3.8
      ? "bg-amber-50 text-amber-800 border-amber-200"
      : "bg-rose-50 text-rose-800 border-rose-200";
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-sans font-700 border rounded-sm ${color}`}
    >
      ★ {rating.toFixed(2)}
    </span>
  );
}

export default function BookCard({ book, size, rank }: Props) {
  const [imgError, setImgError] = useState(false);

  if (size === "large") {
    return (
      <div
        className="
          flex gap-6 bg-cream rounded-sm shadow-book
          border-l-4 border-l-burgundy
          p-5 animate-fade-up
        "
      >
        {/* Cover */}
        <div className="flex-shrink-0">
          {book.thumbnail && !imgError ? (
            <img
              src={book.thumbnail}
              alt={book.title}
              width={140}
              height={210}
              onError={() => setImgError(true)}
              className="rounded-sm"
              style={{
                width: 140,
                height: 210,
                objectFit: "cover",
                boxShadow: "3px 3px 10px rgba(0,0,0,0.25)",
              }}
            />
          ) : (
            <CoverPlaceholder size="large" />
          )}
        </div>

        {/* Info */}
        <div className="flex flex-col gap-2 min-w-0">
          <p className="font-sans text-xs uppercase tracking-widest text-burgundy font-700 opacity-80">
            You searched for
          </p>
          <h2 className="font-serif text-2xl leading-tight text-ink">
            {book.title}
          </h2>
          <p className="font-body text-ink-light italic text-base">
            {book.authors}
          </p>

          <div className="flex flex-wrap gap-2 mt-1">
            <span className="inline-block px-2 py-0.5 bg-parchment border border-parchment-dark text-xs font-sans text-ink-light rounded-sm">
              {book.categories}
            </span>
            {book.average_rating !== null && (
              <RatingBadge rating={book.average_rating} />
            )}
            {book.published_year !== null && (
              <span className="inline-block px-2 py-0.5 bg-parchment border border-parchment-dark text-xs font-sans text-ink-light rounded-sm">
                {book.published_year}
              </span>
            )}
          </div>

          <p className="font-body text-sm text-ink-light leading-relaxed mt-1 line-clamp-4">
            {book.description}
          </p>
        </div>
      </div>
    );
  }

  // Small card (recommendation grid)
  return (
    <div
      className="
        group flex flex-col bg-cream rounded-sm shadow-book
        hover:shadow-book-hover hover:-translate-y-1
        transition-all duration-200 cursor-default overflow-hidden
      "
    >
      {/* Cover */}
      <div className="relative bg-parchment-dark">
        {book.thumbnail && !imgError ? (
          <img
            src={book.thumbnail}
            alt={book.title}
            width={90}
            height={135}
            onError={() => setImgError(true)}
            className="w-full"
            style={{ aspectRatio: "2/3", objectFit: "cover", display: "block" }}
          />
        ) : (
          <div
            style={{ aspectRatio: "2/3" }}
            className="w-full flex items-center justify-center bg-parchment-dark"
          >
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#b89070"
              strokeWidth="1.5"
            >
              <path d="M4 19.5A2.5 2.5 0 016.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
            </svg>
          </div>
        )}
        {rank !== undefined && (
          <span className="absolute top-2 left-2 w-6 h-6 bg-burgundy text-cream text-xs font-sans font-700 rounded-full flex items-center justify-center shadow">
            {rank}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-3 flex flex-col gap-1 flex-1">
        <h3 className="font-serif text-sm leading-snug text-ink line-clamp-2">
          {book.title}
        </h3>
        <p className="font-body text-xs text-ink-light italic line-clamp-1">
          {book.authors}
        </p>
        <div className="flex flex-wrap gap-1 mt-auto pt-2">
          {book.average_rating !== null && (
            <span className="text-xs font-sans text-ink-light opacity-70">
              ★ {book.average_rating.toFixed(1)}
            </span>
          )}
          {book.published_year !== null && (
            <span className="text-xs font-sans text-ink-light opacity-50 ml-auto">
              {book.published_year}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
