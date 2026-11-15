# HELIX-Serve

[![CI](https://github.com/kapildhinwa2010-dev/HELIX-Serve/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/HELIX-Serve/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/HELIX-Serve/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/HELIX-Serve/actions/workflows/codeql.yml)

Serve wave-lattice recall requests: pattern matching with hit/miss statistics.

## What it does

MemoryService stores unit-normalized pattern vectors and serves recall
requests: each probe is matched to the most similar stored pattern by
dot product, weak matches are rejected below a threshold, and every
serve call records latency plus hit/miss statistics.

    from helix_serve import MemoryService

    service = MemoryService(dim=32, threshold=0.5)
    service.add("pattern-a", vec_a)
    out = service.serve("req-1", probe)
    print(out["hit"], out.get("name"), out["latency_ms"])

## API

- MemoryService(dim, threshold) - add / recall / serve / stats
- RecallMiss - raised when no pattern is similar enough
- ServeRegistry - the generic base registry (unchanged from DNA-Serve)

## Honest results and limitations

- Similarity is a plain dot product on the features you store; feature
  choice decides everything and this library does not learn features.
- Recall is a linear scan over stored patterns.
- This is an in-process service layer, not a network server; latency
  figures are in-process function-call times.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
