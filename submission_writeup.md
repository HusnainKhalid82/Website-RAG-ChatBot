# Submission Write-up

## Chunking and retrieval strategy

I chose a chunk size of 1000 characters with 200 characters of overlap. This gives each chunk enough context for the LLM to answer grounded questions while keeping the vector search precise. Overlap helps preserve continuity across paragraph boundaries and reduces the chance that a relevant sentence is split from its supporting context.

For retrieval, I used Chroma with a top-k similarity search. The vector store returns the most semantically relevant chunks for the question, and the LangGraph flow uses those chunks to judge relevance before generating an answer.

## Where the bot fails

1. If the website content is highly structured or contains many short snippets, the fixed chunk size can produce noisy matches.
2. Follow-up questions can still fail when the history is too short or the question is too vague after context stripping.
3. The bot may struggle with websites that require JavaScript rendering or dynamic loading, since the current loader only fetches raw HTML.

## Production improvements

1. Add hybrid search: combine sparse BM25 with vector search to improve precision on keyword-heavy pages.
2. Use a more advanced website ingestion pipeline that strips boilerplate with an HTML cleaner and supports JavaScript-rendered sites.
3. Add automated tests using a small Q&A dataset and measure retrieval/answer quality.
4. Add logging, metrics, and better cache invalidation for vector store updates.
