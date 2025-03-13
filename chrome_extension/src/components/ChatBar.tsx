import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  IconButton,
  Snackbar,
  TextField,
  Tooltip,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import SendIcon from "@mui/icons-material/Send";
import DeleteForeverIcon from "@mui/icons-material/DeleteForever";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import {
  Avatar,
  ChatContainer,
  Message,
  MessageList,
  TypingIndicator,
} from "@chatscope/chat-ui-kit-react";
import Markdown from "markdown-to-jsx";
import {
  DEFAULT_HOST,
  apiConnected,
  getChatOptions,
} from "../scripts/options";
import { getHtmlContent } from "../scripts/content";
import { CodeBlock, PreBlock } from "./CodeBlock";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import "./ChatBar.css";

export class ChatMessage {
  constructor(
    public sender: string,
    public message: string,
  ) {}
}

const ChatBar: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [promptError, setPromptError] = useState(false);
  const [promptPlaceholderText, setPromptPlaceholderText] = useState(
    "Enter your prompt here",
  );
  const [parsingDisabled, setParsingDisabled] = useState(false);
  const [highlightedContent, setHighlightedContent] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [submitDisabled, setSubmitDisabled] = useState(false);
  const [loading1, setLoading1] = useState(false); // loading state during embedding process
  const [loading1Text, setLoading1Text] = useState("");
  const [loading2, setLoading2] = useState(false); // loading state during completion process
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const textFieldRef = useRef<HTMLInputElement | null>(null);
  const [chatContainerHeight, setChatContainerHeight] = useState(300);
  const [currentChatId, setCurrentChatId] = useState("");

  const handlePromptChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPrompt(event.target.value);
    chrome.storage.session.set({ prompt: event.target.value });
  };

  const saveMessages = (messages: ChatMessage[]) => {
    setMessages(messages);
    chrome.storage.session.set({ messages: messages });
  };

  const saveCurrentChatId = (chatId: string) => {
    setCurrentChatId(chatId);
    chrome.storage.session.set({ currentChatId: chatId });
  };

  const updateChat = (chatId?: string) => {
    if (chatId) {
      chrome.storage.local.get(["chatHistory"], (data) => {
        if (data.chatHistory) {
          // chat history already exists in local storage
          const newChatHistory = data.chatHistory;
          // add new chat to chat history
          newChatHistory[chatId] = {
            updatedAt: Date.now(),
            preview: messages[0].message,
            messages: messages,
          };

          // save new chat history in local storage and update current chat ID
          saveCurrentChatId(chatId);
          chrome.storage.local.set({ chatHistory: newChatHistory });
        }
      });
    }
  };

  const promptWithContent = async () => {
    // get default options
    const options = getChatOptions();
    const contentConfig = options.contentConfig;
    const config = contentConfig["default"];
    let activeTabUrl: URL;

    chrome.tabs
      .query({ active: true, currentWindow: true })
      .then((tabs) => {
        const activeTab = tabs[0];
        const activeTabId = activeTab.id || 0;
        activeTabUrl = new URL(activeTab.url || "");

        if (activeTabUrl.protocol === "chrome:") {
          // skip script injection for chrome:// urls
          const result = new Array(1);
          result[0] = { result: [prompt, false, []] };
          return result;
        } else {
          return chrome.scripting.executeScript({
            target: { tabId: activeTabId },
            injectImmediately: true,
            func: getHtmlContent,
            args: [config.selectors, config.selectorsAll],
          });
        }
      })
      .then(async (results) => {
        const pageContent = results[0].result[0];
        const isHighlightedContent = results[0].result[1];
        const imageURLs = results[0].result[2];

        setHighlightedContent(isHighlightedContent);

        chrome.runtime.sendMessage({ context: pageContent }).then(() => {
          chrome.runtime.sendMessage({
            prompt: prompt,
            skipRAG: false,
            chunkSize: config.chunkSize,
            chunkOverlap: config.chunkOverlap,
            url: activeTabUrl.toString(),
            skipCache: isHighlightedContent,
            imageURLs: imageURLs,
          });
        });
      })
      .catch((error) => {
        console.log(`Error: ${error}`);
      });
  };

  const handleSendButtonClick = async () => {
    setLoading1(true);
    setLoading1Text("Submitting question...");
    setSubmitDisabled(true);

    // save user message to messages list
    const newMessages = [...messages, new ChatMessage("user", prompt)];
    saveMessages(newMessages);

    if (parsingDisabled) {
      chrome.runtime.sendMessage({ prompt: prompt, skipRAG: true });
    } else {
      promptWithContent();
    }

    // clear prompt after sending it to the background script
    setPrompt("");
    chrome.storage.session.set({ prompt: "" });
  };

  const handleAvatarClick = (message: string) => {
    navigator.clipboard.writeText(message);
    setShowSnackbar(true);
    setSnackbarMessage("Copied!");
  };

  const handleClearButtonClick = () => {
    saveMessages([]);
    saveCurrentChatId("");
  };

  const appendNonUserMessage = (
    currentMessages: ChatMessage[],
    sender: string,
    completion: string,
  ): ChatMessage[] => {
    const newMsg = new ChatMessage(sender, completion);
    const lastMessage = currentMessages[currentMessages.length - 1];
    let newMessages;

    if (lastMessage !== undefined && lastMessage.sender === "user") {
      // append assistant/tool message to messages list
      newMessages = [...currentMessages, newMsg];
    } else {
      // replace last assistant/tool message with updated message
      newMessages = [
        ...currentMessages.slice(0, currentMessages.length - 1),
        newMsg,
      ];
    }

    setMessages(newMessages);
    return newMessages;
  };

  const handleBackgroundMessage = (msg: {
    docNo: number;
    docCount: number;
    skipCache: boolean;
    completion: string;
    sender: string;
    done: boolean;
  }) => {
    if (msg.docNo) {
      const skipCacheMsg = msg.skipCache ? " (skipping cache)" : "";
      setLoading1(true);
      setLoading1Text(
        `Generating embeddings ${msg.docNo} of ${msg.docCount}${skipCacheMsg}`,
      );
    } else if (msg.completion) {
      setLoading1(false);
      setLoading2(true);
      appendNonUserMessage(messages, msg.sender, msg.completion);
    } else if (msg.done) {
      // save messages after response streaming is done
      chrome.storage.session.set({ messages: messages });
      setLoading2(false);
      setSubmitDisabled(false);
      updateChat(currentChatId);
    }
  };

  useEffect(() => {
    chrome.runtime.onMessage.addListener(handleBackgroundMessage);

    return () => {
      chrome.runtime.onMessage.removeListener(handleBackgroundMessage);
    };
  });

  useEffect(() => {
    chrome.storage.local.get(
      ["chatContainerHeight", "selectedModel", "selectedHost", "chatHistory"],
      async (data) => {
        if (data.chatContainerHeight) {
          setChatContainerHeight(data.chatContainerHeight);
        }
        if (data.chatHistory) {
          chrome.storage.session.get(["currentChatId"], (sessionData) => {
            if (
              sessionData.currentChatId &&
              data.chatHistory[sessionData.currentChatId]
            ) {
              // Only set the current chat ID if it's present in the chat history.
              // It may have been deleted in the chat history view.
              console.log(
                "Setting current chat id:",
                sessionData.currentChatId,
              );
              setCurrentChatId(sessionData.currentChatId);
            }
          });
        }

        // API connectivity check
        const selectedHost = data.selectedHost || DEFAULT_HOST;
        const [connected, models, errMsg] = await apiConnected(selectedHost);

        if (connected) {
          setPromptError(false);
          setPromptPlaceholderText("Enter your prompt here");

          if (!data.selectedModel) {
            // persist selected model to local storage
            chrome.storage.local.set({ selectedModel: models[0] });
          }
        } else {
          setPromptError(true);
          setPromptPlaceholderText(errMsg);
        }
      },
    );

    chrome.storage.session.get(
      ["prompt", "parsingDisabled", "messages", "currentChatId"],
      (data) => {
        if (data.prompt) {
          setPrompt(data.prompt);
        }
        if (data.parsingDisabled) {
          setParsingDisabled(data.parsingDisabled);
        }
        if (data.messages) {
          const currentMsgs = data.messages;
          setMessages(currentMsgs);

          // check if there is a completion in storage to append to the messages list
          chrome.storage.sync.get(["completion", "sender"], (data) => {
            if (data.completion && data.sender) {
              const newMessages = appendNonUserMessage(
                currentMsgs,
                data.sender,
                data.completion,
              );
              chrome.storage.session.set({ messages: newMessages });

              setLoading2(false);
              setSubmitDisabled(false);
              chrome.storage.sync.remove(["completion", "sender"]);
            }
          });
        }
      },
    );
  }, []);

  useEffect(() => {
    if (!submitDisabled && textFieldRef.current) {
      textFieldRef.current.focus();
    }
  }, [submitDisabled]);

  return (
    <Box>
      <Box className="chat-container" sx={{ height: chatContainerHeight }}>
        <Snackbar
          anchorOrigin={{ vertical: "top", horizontal: "center" }}
          open={showSnackbar}
          autoHideDuration={1500}
          onClose={() => setShowSnackbar(false)}
        >
          <Alert
            onClose={() => setShowSnackbar(false)}
            severity="success"
            sx={{ width: "100%" }}
          >
            {snackbarMessage}
          </Alert>
        </Snackbar>
        <ChatContainer>
          <MessageList
            typingIndicator={
              (loading1 || loading2) && (
                <TypingIndicator
                  content={loading1 ? loading1Text : "Generating answers!"}
                />
              )
            }
          >
            {messages.map((message, index) => (
              <Message
                key={index}
                model={{
                  sender: message.sender,
                  direction:
                    message.sender === "user" ? "outgoing" : "incoming",
                  position: "single",
                }}
                type="custom"
              >
                <Avatar 
                onClick={() => handleAvatarClick(message.message)}
                >
                  <ContentCopyIcon color="primary"/>
                </Avatar>
                <Message.CustomContent>
                  <Markdown
                    options={{
                      overrides: {
                        pre: PreBlock,
                        code: CodeBlock,
                      },
                    }}
                  >
                    {message.message.trim()}
                  </Markdown>
                </Message.CustomContent>
              </Message>
            ))}
          </MessageList>
        </ChatContainer>
      </Box>
      <Box sx={{ display: "flex", alignItems: "center" }}>
        {highlightedContent && (
          <Tooltip title="Page has highlighted content" placement="top">
            <InfoIcon fontSize="small" color="primary" />
          </Tooltip>
        )}
      </Box>
      <Box className="chat-bar">
        <TextField
          className="input-field"
          multiline
          maxRows={5}
          placeholder={promptPlaceholderText}
          value={prompt}
          disabled={submitDisabled}
          error={promptError}
          onChange={handlePromptChange}
          inputRef={textFieldRef}
          onKeyUp={(event) => {
            if (!event.shiftKey && event.key === "Enter") {
              handleSendButtonClick();
            }
          }}
          sx={{
            "& .MuiInputBase-root.Mui-error": {
              WebkitTextFillColor: "red",
            },
          }}
        />
        <Tooltip title="Submit question">
          <IconButton
            className="submit-button"
            disabled={submitDisabled || prompt === ""}
            onClick={handleSendButtonClick}
          >
            <SendIcon color="primary"/>
          </IconButton>
        </Tooltip>
        <Tooltip title="Clear history">
          <IconButton
            className="clear-button"
            disabled={submitDisabled}
            onClick={handleClearButtonClick}
          >
            <DeleteForeverIcon color="primary"/>
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default ChatBar;
