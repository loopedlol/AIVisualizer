# AI Visualizer

## Summary

This project generates vectors from text by embedding them using HuggingFace, and then uses these vector databases to search for similarities in passages. The code takes a text file, chunks the contents, embed the chunks, and then formats them into a vector database file. This allows the user to find similar passages in a unique text.

## Files

- `random_passage.txt` - Sample passage used by the program
- `text_embedder.py` - Takes text chunks from the passage and embeds them into vector databases
- `search_engine.py` - Uses the vector database to find similar results to the query