import re
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Data structures ───────────────────────────────────────────────────────────


class Chunk:
    def __init__(self, file_path: str, content: str, chunk_id: int = 0, file_type: str = ""):
        self.file_path = file_path
        self.content = content
        self.chunk_id = chunk_id
        self.file_type = file_type or _categorize_file(file_path)


# ── File type categorization ──────────────────────────────────────────────────


def _categorize_file(path: str) -> str:
    p = path.lower()
    if p.endswith(".sql"):
        return "Database migration"
    if "/api/" in p and p.endswith((".ts", ".js")):
        return "API route handler"
    if p.endswith(".tsx") and "/components/" in p:
        return "React component"
    if p.endswith(".tsx") and "/app/" in p:
        return "Page / layout"
    if "/lib/" in p:
        return "Library module"
    if "/hooks/" in p:
        return "React hook"
    if "/types/" in p:
        return "Type definitions"
    if p.endswith(".md"):
        return "Documentation"
    if p.endswith(".css"):
        return "Stylesheet"
    if p.endswith(".json"):
        return "Configuration"
    if p.endswith((".ts", ".js")):
        return "Source module"
    return "Project file"


# ── Tokenizer ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> List[str]:
    """Code-aware tokenizer: splits camelCase, snake_case, paths, kebab-case."""
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_./\-]*|[0-9]+", text)
    tokens = []
    for tok in raw:
        lowered = tok.lower()
        tokens.append(lowered)
        if "/" in tok or "." in tok:
            tokens.extend(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lowered))
        if "_" in tok:
            tokens.extend(p for p in lowered.split("_") if len(p) > 1)
        if "-" in tok:
            tokens.extend(p for p in lowered.split("-") if len(p) > 1)
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+", tok)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if len(p) > 1)
    return tokens


# ── Query expansion ───────────────────────────────────────────────────────────

SYNONYMS = {
    "tag": ["classify", "classification", "category", "label", "categorize"],
    "tagging": ["classify", "classification", "categorize", "category", "label"],
    "label": ["classify", "classification", "category", "tag"],
    "sort": ["classify", "classification", "filter", "category"],
    "score": ["scoring", "rank", "ranking", "rate", "rating", "carrier"],
    "scoring": ["score", "rank", "ranking", "carrier", "weight"],
    "send": ["blast", "outbound", "dispatch", "compose", "email"],
    "blast": ["send", "outbound", "carrier", "email", "blast"],
    "login": ["auth", "authentication", "signin", "middleware", "session"],
    "auth": ["login", "authentication", "signin", "middleware", "session", "supabase"],
    "signup": ["auth", "register", "registration"],
    "db": ["database", "schema", "migration", "supabase", "sql", "table"],
    "database": ["schema", "migration", "supabase", "sql", "table", "column"],
    "schema": ["database", "migration", "sql", "table", "column", "type"],
    "ai": ["openai", "gpt", "llm", "prompt", "classify", "agent", "claude"],
    "agent": ["ai", "autonomous", "orchestrator", "pipeline", "cycle"],
    "email": ["inbox", "outlook", "message", "sync", "graph"],
    "inbox": ["email", "outlook", "message", "sync"],
    "load": ["shipment", "freight", "tender", "order", "lane"],
    "carrier": ["trucker", "driver", "transport", "scoring", "blast"],
    "shipper": ["customer", "client", "tender", "load"],
    "route": ["api", "endpoint", "handler", "request"],
    "endpoint": ["api", "route", "handler"],
    "component": ["ui", "react", "tsx", "page", "render"],
    "hook": ["useEffect", "useState", "react", "custom"],
    "negotiate": ["negotiation", "rate", "counter", "offer", "accept"],
    "rate": ["price", "cost", "negotiate", "quote", "bid"],
    "quote": ["bid", "rate", "price", "board", "offer"],
    "settings": ["config", "configuration", "preferences", "options"],
    "deploy": ["vercel", "build", "hosting", "production"],
    "test": ["testing", "spec", "jest", "check"],
    "error": ["bug", "issue", "fix", "problem", "fail"],
    "realtime": ["supabase", "subscription", "websocket", "live", "sync"],
    "pdf": ["attachment", "document", "extract", "parse"],
    "ttms": ["tms", "transport", "management", "system", "integration"],
}


def expand_query(query: str) -> str:
    words = set(re.findall(r"[a-zA-Z]+", query.lower()))
    extra = set()
    for w in words:
        if w in SYNONYMS:
            extra.update(SYNONYMS[w])
    return query + " " + " ".join(extra - words)


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_ingest_file(file_path: str) -> Tuple[str, List[Chunk]]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    separator = "=" * 48
    first_sep = content.find(separator)
    dir_tree = content[:first_sep].strip() if first_sep > 0 else ""

    pattern = re.compile(
        rf"{separator}\nFILE: (.+?)\n{separator}\n(.*?)(?=\n{separator}\nFILE:|$)",
        re.DOTALL,
    )

    chunks = []
    for match in pattern.finditer(content):
        fpath = match.group(1).strip()
        fcontent = match.group(2).strip()
        if not fcontent:
            continue

        lines = fcontent.split("\n")
        max_lines = 120
        overlap = 30

        if len(lines) <= max_lines:
            chunks.append(Chunk(fpath, fcontent))
        else:
            # Grab imports/header from the first ~15 lines for context propagation
            header_lines = []
            for line in lines[:20]:
                s = line.strip()
                if s.startswith(("import ", "from ", "const ", "'use ", '"use ', "//", "/*", " *", "*")):
                    header_lines.append(line)
                elif s == "":
                    continue
                elif len(header_lines) > 0:
                    break
            import_header = "\n".join(header_lines[:10]) + "\n// ...\n" if header_lines else ""

            for i in range(0, len(lines), max_lines - overlap):
                sub_lines = lines[i : i + max_lines]
                sub = "\n".join(sub_lines)
                if sub.strip():
                    # Prepend imports to non-first sub-chunks
                    if i > 0 and import_header:
                        sub = import_header + sub
                    chunks.append(Chunk(fpath, sub, i // (max_lines - overlap)))

    return dir_tree, chunks


# ── Hybrid retrieval engine ──────────────────────────────────────────────────


class RAGEngine:
    def __init__(self, ingest_path: str):
        self.dir_tree, self.chunks = parse_ingest_file(ingest_path)

        # ---------- BM25 index (path-boosted) ----------
        self.bm25_corpus = []
        for c in self.chunks:
            path_tokens = tokenize(c.file_path)
            type_tokens = tokenize(c.file_type)
            content_tokens = tokenize(c.content)
            self.bm25_corpus.append(path_tokens * 3 + type_tokens * 2 + content_tokens)
        self.bm25 = BM25Okapi(self.bm25_corpus)

        # ---------- TF-IDF index ----------
        # Join file metadata + content as plain text for TF-IDF
        tfidf_docs = []
        for c in self.chunks:
            doc = f"{c.file_path} {c.file_type} {c.file_path} {c.content}"
            tfidf_docs.append(doc)

        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=lambda x: x,
            max_features=60000,
            sublinear_tf=True,
            token_pattern=None,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(tfidf_docs)

    _CODE_EXTS = frozenset((".ts", ".tsx", ".js", ".jsx", ".sql", ".css", ".py"))

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        expanded = expand_query(query)
        original_tokens = tokenize(query)
        expanded_tokens = tokenize(expanded)
        pool_size = top_k * 4

        # ---- Signal 1: BM25 on original query (precise) ----
        bm25_orig = self.bm25.get_scores(original_tokens)
        bm25_orig_ranking = np.argsort(-bm25_orig)[:pool_size]

        # ---- Signal 2: BM25 on expanded query (broad recall) ----
        bm25_exp = self.bm25.get_scores(expanded_tokens)
        bm25_exp_ranking = np.argsort(-bm25_exp)[:pool_size]

        # ---- Signal 3: TF-IDF (semantic similarity) ----
        query_vec = self.tfidf_vectorizer.transform([expanded])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_ranking = np.argsort(-tfidf_scores)[:pool_size]

        # ---- Signal 4: File-path match with expanded terms ----
        path_query_words = set(re.findall(r"[a-zA-Z]{2,}", expanded.lower()))
        path_scores = {}
        for i, c in enumerate(self.chunks):
            path_words = set(re.findall(r"[a-zA-Z]{2,}", c.file_path.lower()))
            overlap = len(path_query_words & path_words)
            if overlap > 0:
                path_scores[i] = overlap
        path_ranking = sorted(path_scores, key=path_scores.get, reverse=True)[:pool_size]

        # ---- Reciprocal Rank Fusion (4 signals) ----
        k = 60
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_orig_ranking):
            rrf[idx] = rrf.get(idx, 0) + 1.2 / (k + rank)   # Slightly higher weight
        for rank, idx in enumerate(bm25_exp_ranking):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank)
        for rank, idx in enumerate(tfidf_ranking):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank)
        for rank, idx in enumerate(path_ranking):
            rrf[idx] = rrf.get(idx, 0) + 0.8 / (k + rank)   # Boosted path signal

        # ---- Boost source code files over docs ----
        for idx in rrf:
            c = self.chunks[idx]
            ext = "." + c.file_path.rsplit(".", 1)[-1] if "." in c.file_path else ""
            if ext in self._CODE_EXTS:
                rrf[idx] *= 1.4  # 40% boost for actual code

        # ---- Select results with per-file diversity ----
        final = sorted(rrf, key=rrf.get, reverse=True)
        max_score = max(bm25_orig.max(), bm25_exp.max(), 1.0)
        results = []
        file_counts: dict[str, int] = {}
        max_per_file = 2

        for idx in final:
            # Filter zero-relevance junk
            if bm25_orig[idx] <= 0 and bm25_exp[idx] <= 0 and tfidf_scores[idx] <= 0.01:
                continue
            chunk = self.chunks[idx]
            count = file_counts.get(chunk.file_path, 0)
            if count >= max_per_file:
                continue
            results.append(chunk)
            file_counts[chunk.file_path] = count + 1
            if len(results) >= top_k:
                break

        return results

    def get_summary(self) -> str:
        """Pre-computed codebase summary for aggregate questions."""
        if hasattr(self, "_summary"):
            return self._summary

        all_paths = list(dict.fromkeys(c.file_path for c in self.chunks))

        api_routes = [p for p in all_paths if "/api/" in p and p.endswith("route.ts")]
        components = [p for p in all_paths if "/components/" in p and p.endswith(".tsx")]
        pages = [p for p in all_paths if "/app/" in p and p.endswith("page.tsx")]
        hooks = [p for p in all_paths if "/hooks/" in p]
        migrations = [p for p in all_paths if "/migrations/" in p and p.endswith(".sql")]
        lib_modules = [p for p in all_paths if "/lib/" in p and p.endswith(".ts")]
        docs = [p for p in all_paths if p.endswith(".md")]
        types = [p for p in all_paths if "/types/" in p]

        lines = [
            f"CODEBASE STATISTICS:",
            f"- Total files: {len(all_paths)}",
            f"- API endpoints (route.ts): {len(api_routes)}",
            f"- React components (.tsx): {len(components)}",
            f"- Pages: {len(pages)}",
            f"- Hooks: {len(hooks)}",
            f"- Database migrations: {len(migrations)}",
            f"- Library modules (src/lib): {len(lib_modules)}",
            f"- Documentation files (.md): {len(docs)}",
            f"- Type definition files: {len(types)}",
            f"",
            f"API ENDPOINTS ({len(api_routes)}):",
        ]
        for r in api_routes:
            # Extract the route path from file path
            # e.g. src/app/api/emails/classify/route.ts -> /api/emails/classify
            route = r.split("/api/")[-1].replace("/route.ts", "")
            lines.append(f"  /api/{route}")

        lines.append(f"")
        lines.append(f"DATABASE MIGRATIONS ({len(migrations)}):")
        for m in migrations:
            lines.append(f"  {m.split('/')[-1]}")

        lines.append(f"")
        lines.append(f"REACT COMPONENTS ({len(components)}):")
        for c in components:
            lines.append(f"  {c.split('/components/')[-1]}")

        self._summary = "\n".join(lines)
        return self._summary

    def get_context(
        self, query: str, top_k: int = 10, max_chars: int = 18000
    ) -> Tuple[str, List[str]]:
        chunks = self.retrieve(query, top_k)
        parts, sources, total = [], [], 0

        for c in chunks:
            header = f"--- FILE: {c.file_path} [{c.file_type}]"
            if c.chunk_id > 0:
                header += f" (part {c.chunk_id + 1})"
            header += " ---"
            block = f"{header}\n{c.content}"

            if total + len(block) > max_chars:
                break
            parts.append(block)
            if c.file_path not in sources:
                sources.append(c.file_path)
            total += len(block)

        return "\n\n".join(parts), sources
