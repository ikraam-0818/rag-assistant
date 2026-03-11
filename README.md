# RAG Assistant

A Node.js and TypeScript-based Retrieval-Augmented Generation (RAG) Artificial Intelligence Assistant. This project leverages the OpenAI API for language modeling and ChromaDB as a vector database to provide intelligent, context-aware responses based on provided documents (specifically PDFs).

## Technologies Used

* **Language**: [TypeScript](https://www.typescriptlang.org/) / [Node.js](https://nodejs.org/)
* **LLM Provider**: [OpenAI API](https://openai.com/)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Document Parsing**: `pdf-parse`, `pdf2json`
* **Environment Management**: `dotenv`

## Project Prerequisites

Ensure you have the following installed on your machine:

- **Node.js**: (Version 18+ recommended)
- **OpenAI API Key**: Obtainable from the OpenAI Platform dashboard.
- **ChromaDB**: Ensure Chroma is running locally or you have access to a remote instance. If running locally, you can start it typically via Docker:
  ```bash
  docker run -p 8000:8000 ghcr.io/chroma-core/chroma:latest
  ```

## Getting Started

### 1. Installation

Clone or download the project folder, then install the dependencies using npm:

```bash
npm install
```

### 2. Environment Configuration

Create a `.env` file in the root directory. You can use any testing or staging `.env` templates provided, or simply create it manually:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Add any other required environment variables based on the project code
```

### 3. Running the Project

Because the project is written in TypeScript, you can run it directly using `ts-node`, or build it into JavaScript.

**To run directly (Development):**
Pass your query in quotes after the command:
```bash
npx ts-node src/index.ts "What is the customer policy"
```

**To compile to JavaScript (Production):**
```bash
npx tsc
node dist/index.js "What is the customer policy"
```

*(Note: The exact entry file might change depending on the codebase structure. Verify that `src/index.ts` or `index.ts` is the current main entry point).*

## Project Structure

* `src/`: Contains the TypeScript source files covering logic for vector storage, embedding generation, LLM communication, etc.
* `data/`: Used for storing documents (like PDFs) to be ingested into the vector DB. (Ignored by Git)
* `chroma_db/`: Default location for local persistent storage of ChromaDB. (Ignored by Git)

## Features (Proposed/Implemented)

* Parsing of PDF documents to extract text.
* Converting chunked text into vector embeddings using OpenAI.
* Inserting the embeddings into the Chroma vector database.
* Querying ChromaDB with a user prompt to retrieve the nearest matching context.
* Injecting that context to an OpenAI Chat Completion prompt for enriched, grounded answers.
