import { Document } from "@langchain/core/documents";
import {
  AIMessage,
  BaseMessage,
  HumanMessage
} from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ConsoleCallbackHandler } from "@langchain/core/tracers/console";
import { IterableReadableStream } from "@langchain/core/utils/stream";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ChatMessage } from "../components/ChatBar";
import { EnhancedMemoryVectorStore } from "../vectorstores/enhanced_memory";
import {
  DEFAULT_KEEP_ALIVE,
  getChatOptions,
  ChatOptions,
} from "./options";

interface VectorStoreMetadata {
  vectorStore: EnhancedMemoryVectorStore;
  createdAt: number;
}

// map of url to vector store metadata
const vectorStoreMap = new Map<string, VectorStoreMetadata>();

// global variables
let context = "";
let completion = "";

const MAX_CHAT_HISTORY = 3;
const SYS_PROMPT_TEMPLATE = `Use the following context when responding to the prompt.\n\nBEGIN CONTEXT\n\n{filtered_context}\n\nEND CONTEXT`;

const getChatModel = (options: ChatOptions): ChatOllama => {
  return new ChatOllama({
    baseUrl: options.ollamaHost,
    model: options.ollamaModel,
    keepAlive: DEFAULT_KEEP_ALIVE,
    callbacks: [new ConsoleCallbackHandler()],
  });
};

const getMessages = async(): Promise<BaseMessage[]> => {
  let messages: BaseMessage[] = [];
  // the array of persisted messages includes the current prompt
  const data = await chrome.storage.session.get(["messages"]);

  if (data.messages) {
    const chatMsgs = data.messages as ChatMessage[];
    messages = chatMsgs
      .slice(-1 * MAX_CHAT_HISTORY)
      .map((msg: ChatMessage) => {
        return msg.sender === "user"
          ? new HumanMessage({
              content: msg.message,
            })
          : new AIMessage({
              content: msg.message,
            });
      });
  }
  return messages;
};

const computeK = (documentsCount: number): number => {
  return Math.ceil(Math.sqrt(documentsCount));
};

const streamChunks = async (stream: IterableReadableStream<string>) => {
  completion = "";
  for await (const chunk of stream) {
    completion += chunk;
    chrome.runtime
      .sendMessage({ completion: completion, sender: "assistant" })
      .catch(() => {
        console.log("Sending partial completion, but popup is closed...");
      });
  }
  chrome.runtime.sendMessage({ done: true }).catch(() => {
    console.log("Sending done message, but popup is closed...");
    chrome.storage.sync.set({ completion: completion, sender: "assistant" });
  });
};

chrome.runtime.onMessage.addListener(async (request) => {
  // process prompt (RAG disabled)
  if (request.prompt && request.skipRAG) {
    const prompt = request.prompt.trim();
    console.log(`Received prompt (RAG disabled): ${prompt}`);

    // get options
    const options = getChatOptions();

    // create chain
    const chatPrompt = ChatPromptTemplate.fromMessages(await getMessages());
    const model = getChatModel(options);
    const chain = chatPrompt.pipe(model).pipe(new StringOutputParser());

    // stream response chunks
    const stream = await chain.stream({});
    streamChunks(stream);
  }

  // process prompt (RAG enabled)
  if (request.prompt && !request.skipRAG) {
    const prompt = request.prompt.trim();
    const url = request.url;
    const skipCache = Boolean(request.skipCache);
    console.log(`Received prompt (RAG enabled): ${prompt}`);
    console.log(`Received url: ${url}`);

    // get default content config
    const options = getChatOptions();
    const config = options.contentConfig["default"];
    const chunkSize = request.chunkSize ? request.chunkSize : config.chunkSize;
    const chunkOverlap = request.chunkOverlap
      ? request.chunkOverlap
      : config.chunkOverlap;
    console.log(
      `Received chunk size: ${chunkSize} and chunk overlap: ${chunkOverlap}`,
    );

    // delete all vector stores that are expired
    vectorStoreMap.forEach(
      (vectorStoreMetdata: VectorStoreMetadata, url: string) => {
        if (
          Date.now() - vectorStoreMetdata.createdAt >
          options.vectorStoreTTLMins * 60 * 1000
        ) {
          vectorStoreMap.delete(url);
          console.log(`Deleting vector store for url: ${url}`);
        }
      },
    );

    // check if vector store already exists for url
    let vectorStore: EnhancedMemoryVectorStore;
    let documentsCount: number;

    if (!skipCache && vectorStoreMap.has(url)) {
      // retrieve existing vector store
      console.log(`Retrieving existing vector store for url: ${url}`);
      // eslint-disable-next-line @typescript-eslint/no-non-null-asserted-optional-chain, @typescript-eslint/no-non-null-assertion
      vectorStore = vectorStoreMap.get(url)?.vectorStore!;
      documentsCount = vectorStore.memoryVectors.length;
    } else {
      // create new vector store
      console.log(
        `Creating ${skipCache ? "temporary" : "new"} vector store for url: ${url}`,
      );

      // split page content into overlapping documents
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: chunkSize,
        chunkOverlap: chunkOverlap,
      });
      const documents = await splitter.createDocuments([context]);
      documentsCount = documents.length;

      // load documents into vector store
      vectorStore = new EnhancedMemoryVectorStore(
        new OllamaEmbeddings({
          baseUrl: options.ollamaHost,
          model: options.ollamaModel,
          keepAlive: DEFAULT_KEEP_ALIVE,
        }),
      );
      documents.forEach(async (doc, index) => {
        await vectorStore.addDocuments([
          new Document({
            pageContent: doc.pageContent,
            metadata: { ...doc.metadata, docId: index }, // add document ID
          }),
        ]);
        chrome.runtime
          .sendMessage({
            docNo: index + 1,
            docCount: documentsCount,
            skipCache: skipCache,
          })
          .catch(() => {
            console.log(
              "Sending document embedding message, but popup is closed...",
            );
          });
      });

      // store vector store in vector store map
      if (!skipCache) {
        vectorStoreMap.set(url, {
          vectorStore: vectorStore,
          createdAt: Date.now(),
        });
      }
    }

    // create chain
    const retriever = vectorStore.asRetriever({
      k: computeK(documentsCount),
      searchType: "hybrid",
      callbacks: [new ConsoleCallbackHandler()],
    });

    const chatPrompt = ChatPromptTemplate.fromMessages([
      SystemMessagePromptTemplate.fromTemplate(SYS_PROMPT_TEMPLATE),
      ...(await getMessages()),
    ]);

    const model = getChatModel(options);
    const chain = RunnableSequence.from([
      {
        filtered_context: retriever.pipe(formatDocumentsAsString),
      },
      chatPrompt,
      model,
      new StringOutputParser(),
    ]);

    // stream response chunks
    const stream = await chain.stream(prompt);
    streamChunks(stream);
  }

  // process parsed context
  if (request.context) {
    context = request.context;
    console.log(`Received context: ${context}`);
  }
});

const keepAlive = () => {
  setInterval(chrome.runtime.getPlatformInfo, 20e3);
  console.log("Keep alive...");
};
chrome.runtime.onStartup.addListener(keepAlive);
keepAlive();
