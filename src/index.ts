import * as fs from 'fs';
import * as path from 'path';
import 'dotenv/config';
import OpenAI from 'openai';
import { ChromaClient } from 'chromadb';
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

const COLLECTION_NAME = "rag_knowledge_collection";

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

        // 5. Store in ChromaDB
        console.log("Connecting to ChromaDB and storing chunks...");

        // We try to delete the collection first if it exists to keep this script idempotent
        try {
            await chroma.deleteCollection({ name: COLLECTION_NAME });
        } catch (e) { /* ignore if doesn't exist */ }

        const collection = await chroma.createCollection({
            name: COLLECTION_NAME,
        });

        const ids = chunks.map((_, i) => `chunk_${i}`);

        await collection.add({
            ids: ids,
            embeddings: embeddings,
            documents: chunks,
        });

        console.log("Ingestion complete!");

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
        console.log("Retrieving relevant context...");
        const results = await collection.query({
            queryEmbeddings: [questionVector],
            nResults: 2, // Top 2 chunks
        });

        const retrievedContext = results.documents[0].join("\n\n");
        console.log("Context retrieved:\n", retrievedContext, "\n");

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
