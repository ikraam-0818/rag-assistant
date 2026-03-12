import * as fs from 'fs';
import * as path from 'path';
import 'dotenv/config';
import OpenAI from 'openai';
import { ChromaClient } from 'chromadb';
import { QdrantClient } from '@qdrant/js-client-rest';
const PDFParser = require("pdf2json");
// 1. Initialize Clients
if (!process.env.OPENAI_API_KEY) {
    console.error("Please set OPENAI_API_KEY in your .env file or environment.");
    process.exit(1);
}

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const chroma = new ChromaClient({
    // Default connects to localhost:8000
    // You need a chroma server running for this to work natively (e.g., docker run -p 8000:8000 chromadb/chroma)
});

// We provide our own embeddings via OpenAI. 
// However, Chroma still expects a default embedding function object for collection operations.
// This dummy object satisfies its interface and prevents the noisy warnings.
const dummyEmbeddingFunction = {
    generate: async (texts: string[]) => {
        // Return an empty array of arrays to satisfy the type signature
        return texts.map(() => []); 
    }
};

const qdrant = new QdrantClient({ url: 'http://localhost:6333' });

const COLLECTION_NAME = "rag_knowledge_collection";
const VECTOR_DB = process.env.VECTOR_DB || "chroma"; // "chroma", "qdrant", or "evaluate"

/**
 * A simple function to chunk text into paragraphs or smaller segments
 */
function chunkText(text: string, maxWords: number = 100): string[] {
    const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
    const chunks: string[] = [];

    for (const p of paragraphs) {
        const words = p.split(/\s+/);
        if (words.length <= maxWords) {
            chunks.push(p);
        } else {
            // Very basic sub-chunking
            for (let i = 0; i < words.length; i += maxWords) {
                chunks.push(words.slice(i, i + maxWords).join(" "));
            }
        }
    }
    return chunks;
}

async function main() {
    try {
        console.log("Starting RAG Pipeline...");

        // 2. Load Documents
        const dataDir = path.join(__dirname, '../data');
        const files = fs.readdirSync(dataDir);
        let allText = '';

        for (const file of files) {
            const filePath = path.join(dataDir, file);

            if (file.endsWith('.txt')) {
                console.log(`Reading text file: ${file}`);
                const textData = fs.readFileSync(filePath, 'utf-8');
                allText += textData + '\n\n';
            } else if (file.endsWith('.pdf')) {
                console.log(`Extracting text from PDF: ${file}`);
                const pdfData = await new Promise<string>((resolve, reject) => {
                    const pdfParser = new PDFParser(null, 1);
                    pdfParser.on("pdfParser_dataError", (errData: any) => reject(errData.parserError));
                    pdfParser.on("pdfParser_dataReady", () => resolve(pdfParser.getRawTextContent()));
                    pdfParser.loadPDF(filePath);
                });
                allText += pdfData + '\n\n';
            }
        }

        if (!allText.trim()) {
            console.error("No text could be found or extracted from the data directory.");
            process.exit(1);
        }

        // 3. Chunk Documents
        console.log("Chunking documents...");
        const chunks = chunkText(allText);

        // 4. Create Embeddings
        console.log(`Generating embeddings for ${chunks.length} chunks...`);
        const embeddingResponse = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: chunks,
        });
        const embeddings = embeddingResponse.data.map(d => d.embedding);

        // 5. Store in VectorDB
        console.log(`Connecting to ${VECTOR_DB.toUpperCase()} and storing chunks...`);

        if (VECTOR_DB === 'chroma' || VECTOR_DB === 'evaluate') {
            const startStr = Date.now();
            // We try to delete the collection first if it exists to keep this script idempotent
            try {
                // @ts-ignore - Supress TS complaining because the typings for deleteCollection don't officially support passing embeddingFunction right now
                await chroma.deleteCollection({ name: COLLECTION_NAME, embeddingFunction: dummyEmbeddingFunction });
            } catch (e) { /* ignore if doesn't exist */ }

            const collection = await chroma.createCollection({
                name: COLLECTION_NAME,
                embeddingFunction: dummyEmbeddingFunction
            });

            const ids = chunks.map((_, i) => `chunk_${i}`);

            await collection.add({
                ids: ids,
                embeddings: embeddings,
                documents: chunks,
            });
            const endStr = Date.now();
            console.log(`[Metrics] ChromaDB Ingestion Time: ${endStr - startStr}ms`);
        }
        
        if (VECTOR_DB === 'qdrant' || VECTOR_DB === 'evaluate') {
            const startStr = Date.now();
            try {
                await qdrant.deleteCollection(COLLECTION_NAME);
            } catch (e) { /* ignore */ }

            await qdrant.createCollection(COLLECTION_NAME, {
                vectors: {
                    size: embeddings[0].length, // 1536 for text-embedding-3-small
                    distance: 'Cosine',
                },
            });

            const points = chunks.map((chunk, i) => ({
                id: i + 1, // Qdrant requires numeric or UUID ids
                vector: embeddings[i],
                payload: { document: chunk }
            }));

            await qdrant.upsert(COLLECTION_NAME, {
                wait: true,
                points: points
            });
            const endStr = Date.now();
            console.log(`[Metrics] Qdrant Ingestion Time: ${endStr - startStr}ms`);
        }

        console.log(`Ingestion phase complete!`);

        // ==========================================
        // Query Phase
        // ==========================================

        const question = process.argv.slice(2).join(" ") || "What is a Vector Database used for in RAG?";
        console.log(`\nQuestion: "${question}"\n`);

        // 1. Embed Question
        const qEmbeddingResponse = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: question,
        });
        const questionVector = qEmbeddingResponse.data[0].embedding;

        // 2. Retrieve relevant chunks
        console.log(`Retrieving relevant context from ${VECTOR_DB.toUpperCase()}...`);
        let retrievedContext = "";

        if (VECTOR_DB === 'chroma' || VECTOR_DB === 'evaluate') {
            const startQ = Date.now();
            const collection = await chroma.getCollection({ 
                name: COLLECTION_NAME,
                embeddingFunction: dummyEmbeddingFunction
            });
            const results = await collection.query({
                queryEmbeddings: [questionVector],
                nResults: 2, // Top 2 chunks
            });
            const endQ = Date.now();
            console.log(`[Metrics] ChromaDB Query Time: ${endQ - startQ}ms`);
            
            // In evaluate mode, we default to passing Chroma's context to the LLM
            retrievedContext = results.documents[0].join("\n\n");
            
            if (VECTOR_DB === 'evaluate') {
                console.log("[EVALUATE] Chroma Retrieved Context:\n", retrievedContext, "\n");
            }
        }
        
        if (VECTOR_DB === 'qdrant' || VECTOR_DB === 'evaluate') {
            const startQ = Date.now();
            const results = await qdrant.search(COLLECTION_NAME, {
                vector: questionVector,
                limit: 2,
            });
            const endQ = Date.now();
            console.log(`[Metrics] Qdrant Query Time: ${endQ - startQ}ms`);
            
            const qdrantContext = results.map(hit => hit.payload?.document as string).join("\n\n");
            
            if (VECTOR_DB === 'evaluate') {
                console.log("[EVALUATE] Qdrant Retrieved Context:\n", qdrantContext, "\n");
            }
            
            if (VECTOR_DB === 'qdrant') {
                retrievedContext = qdrantContext;
            }
        }

        console.log("Context retrieved successfully (Content truncated for brevity)\n");

        // 3. Generate Answer
        console.log("Generating Answer using LLM...");
        const prompt = `Use the following context to answer the user's question. If you cannot answer based on the context, say "I don't know based on the provided context."\n\nContext:\n${retrievedContext}\n\nQuestion: ${question}\n\nAnswer:`;

        const chatResponse = await openai.chat.completions.create({
            model: "gpt-4o-mini", // Cost-effective model
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: prompt }
            ],
            temperature: 0.2, // Low temperature for more factual answers
        });

        console.log("\n=================== FINAL ANSWER ===================");
        console.log(chatResponse.choices[0].message.content);
        console.log("====================================================");

    } catch (error) {
        console.error("Error in RAG Pipeline:", error);
    }
}

main();
