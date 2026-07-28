# Cost Benchmark: Embedded (LanceDB) vs. Managed Vector DB

Assumptions: 384-dim embeddings (float32), ~0.5 KB metadata/vector, $0.08/GB/month disk (EBS gp3-class pricing), managed DB priced on Pinecone-shaped always-on pod tiers. See README Cost Analysis section for the full assumption list.

| Vector Count | Storage (GB) | Embedded Cost/mo | Managed Cost/mo | Savings |
|---:|---:|---:|---:|---:|
| 100,000 | 0.19 | $0.0153 | $70.00 | 100.0% |
| 1,000,000 | 1.91 | $0.1526 | $280.00 | 100.0% |
| 10,000,000 | 19.07 | $1.5259 | $1200.00 | 99.9% |

## LLM Generation Cost (separate from vector storage)

At 50,000 queries/month, ~$0.000131/query (550 avg prompt tokens, 80 avg completion tokens, gpt-4o-mini-class pricing) -> **$6.53/month**.

This is the dominant cost at low-to-moderate vector counts: the vector-store line item stays under a dollar a month through 10M vectors, while LLM generation cost scales with query volume, not corpus size.