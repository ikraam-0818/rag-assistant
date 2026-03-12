# RAG Assistant

A Node.js and TypeScript-based Retrieval-Augmented Generation (RAG) Artificial Intelligence Assistant. This project leverages the OpenAI API for language modeling and ChromaDB as a vector database to provide intelligent, context-aware responses based on provided documents (specifically PDFs).

## Technologies Used

* **Language**: [TypeScript](https://www.typescriptlang.org/) / [Node.js](https://nodejs.org/)
* **LLM Provider**: [OpenAI API](https://openai.com/)
* **Vector Databases**: [ChromaDB](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/)
* **Document Parsing**: `pdf-parse`, `pdf2json`
* **Environment Management**: `dotenv`

## Project Prerequisites

Ensure you have the following installed on your machine:

- **Node.js**: (Version 18+ recommended)
- **OpenAI API Key**: Obtainable from the OpenAI Platform dashboard.
- **Vector Database**: Ensure you have either ChromaDB or Qdrant running locally to store embeddings.
  - **ChromaDB**:
    ```bash
    docker run -p 8000:8000 ghcr.io/chroma-core/chroma:latest
    ```
  - **Qdrant**:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
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
VECTOR_DB=chroma # Use 'chroma' or 'qdrant'
# Add any other required environment variables based on the project code
```

### 3. Running the Project

Because the project is written in TypeScript, you can run it directly using `ts-node`, or build it into JavaScript.
You can dynamically switch between vector databases by modifying `.env` or prefixing the command with `VECTOR_DB`. Options are `chroma`, `qdrant` or `evaluate`.

**To run in Evaluate Mode (Development):**
This mode runs ingestion and queries on both databases side-by-side, outputting time metrics.
```bash
VECTOR_DB=evaluate npx ts-node src/index.ts "What is the customer policy"
```
Pass your query in quotes after the command:
```bash
VECTOR_DB=qdrant npx ts-node src/index.ts "What is the customer policy"
```

**To compile to JavaScript (Production):**
```bash
npx tsc
VECTOR_DB=chroma node dist/index.js "What is the customer policy"
```

*(Note: The exact entry file might change depending on the codebase structure. Verify that `src/index.ts` or `index.ts` is the current main entry point).*

## Project Structure

* `src/`: Contains the TypeScript source files covering logic for vector storage, embedding generation, LLM communication, etc.
* `data/`: Used for storing documents (like PDFs) to be ingested into the vector DB. (Ignored by Git)
* `chroma_db/`: Default location for local persistent storage of ChromaDB. (Ignored by Git)

## Features (Proposed/Implemented)

* Parsing of PDF documents to extract text.
* Converting chunked text into vector embeddings using OpenAI.
* Inserting the embeddings into the vector database (ChromaDB or Qdrant).
* Toggling between vector databases dynamically using environment variables. 
* Querying the database with a user prompt to retrieve the nearest matching context.
* Injecting that context to an OpenAI Chat Completion prompt for enriched, grounded answers.
