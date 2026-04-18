// lib/api.ts
// Typed fetch wrappers for the next-book FastAPI backend.
// In dev: set NEXT_PUBLIC_API_URL=http://localhost:8000
// In prod (same-origin): leave unset (defaults to "")

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL !== undefined
    ? process.env.NEXT_PUBLIC_API_URL
    : "";

export type Book = {
  isbn13: string;
  title: string;
  authors: string;
  categories: string;
  description: string;
  thumbnail: string | null;
  average_rating: number | null;
  published_year: number | null;
};

export type SearchResult = { isbn13: string; title: string };

export type RecommendResponse = {
  query: Book;
  recommendations: Book[];
  model: string;
};

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function searchBooks(q: string, limit = 10): Promise<SearchResult[]> {
  if (q.trim().length < 2) return [];
  const data = await apiFetch<{ results: SearchResult[] }>(
    `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`
  );
  return data.results;
}

export async function getBook(isbn: string): Promise<Book> {
  return apiFetch<Book>(`/api/book/${encodeURIComponent(isbn)}`);
}

export async function getRecommendations(
  isbn: string,
  model: "naive" | "classical" | "deep",
  k = 10
): Promise<RecommendResponse> {
  return apiFetch<RecommendResponse>(
    `/api/recommend?isbn=${encodeURIComponent(isbn)}&model=${model}&k=${k}`
  );
}
