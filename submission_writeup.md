# Submission Write-up

## Chunking and retrieval strategy

I chunk with `RecursiveCharacterTextSplitter` at **chunk_size=1000 characters** and
**chunk_overlap=200**. 1000 characters is roughly a paragraph or two: large enough that a
single chunk usually carries a complete, answerable idea, but small enough that its
embedding stays focused and the top-k results are not diluted by unrelated text. The 200
character overlap (20%) keeps a sentence from being cut off from the context that supports
it, which matters at chunk boundaries. I chose the *recursive* splitter over plain
`CharacterTextSplitter` because it falls back through a list of separators
(`\n\n` -> `\n` -> `. ` -> ` `), so it still produces well-sized chunks even after the HTML
cleaner has collapsed the page into text without blank-line paragraph breaks — the plain
splitter silently emitted one oversized chunk per page.

For retrieval I use Chroma with **top-k=3** similarity search. Before generating, the graph
runs an LLM `grade_documents` step that judges whether those chunks are actually relevant;
if not, it rewrites the query and retries (up to 2 times) before falling back. So retrieval
quality is defended by the graph, not just by k.

## Where the bot fails

1. **JavaScript-rendered sites.** The loader fetches raw HTML, so single-page apps that
   render content client-side (React/Vue) come back nearly empty and the bot has nothing to
   ground on.
2. **Vague follow-ups with no keyword anchor.** "Tell me more about the memory part" works
   because "memory" is a distinct term, but a follow-up like "what about that one?" gives the
   retriever nothing specific to match; the rewrite step helps but cannot always recover the
   intended referent.
3. **Aggregation / cross-page questions.** With top-k=3, a question like "list everything the
   site offers" that requires stitching facts from many pages returns an incomplete answer,
   because only three chunks reach the generator.

## Production improvements

1. **Hybrid search (BM25 + vector).** Combine sparse keyword matching with dense retrieval so
   exact terms, names, and codes are not lost to semantic drift.
2. **Better ingestion.** Add a sitemap/recursive crawler and a headless-browser render step
   (e.g. Playwright) so JS-heavy sites are ingested properly.
3. **History-aware retrieval.** Contextualize the follow-up question against chat history
   *before* the first retrieval, instead of relying on the grade-and-rewrite retry loop to
   recover it — saves a round-trip and improves follow-up recall.
4. **Evaluation harness.** A labeled Q&A set with retrieval-hit and answer-faithfulness
   metrics, run in CI, so regressions in chunking/prompts are caught automatically.
5. **Durable state + observability.** Persist conversation state with a LangGraph
   checkpointer and add tracing/metrics for latency, retries, and fallback rate.
